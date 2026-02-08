import chromadb
from chromadb.config import Settings
from openai import OpenAI


class FinancialSituationMemory:
    def __init__(self, name, config):
        self.config = config
        self.embedding_provider = (config.get("embedding_provider") or "").lower()
        self.embedding_model = config.get("embedding_model")
        
        # Determine embedding provider if not explicitly set
        if not self.embedding_provider:
            if config.get("backend_url") == "http://localhost:11434/v1":
                self.embedding_provider = "ollama"
            else:
                deep_model = config.get("deep_think_llm", "")
                quick_model = config.get("quick_think_llm", "")
                using_gemini = (
                    deep_model.startswith(("gemini", "google")) or
                    quick_model.startswith(("gemini", "google"))
                )
                self.embedding_provider = "openai" if using_gemini else "openai"

        # Initialize embedding client based on provider
        if self.embedding_provider == "ollama":
            self.embedding = self.embedding_model or "nomic-embed-text"
            self.embedding_client = OpenAI(base_url=config["backend_url"])
        elif self.embedding_provider == "google":
            if not config.get("google_api_key"):
                raise ValueError(
                    "GOOGLE_API_KEY is required when embedding_provider is set to 'google'."
                )
            # Use the current Gemini embedding model by default (text-embedding-004 is deprecated)
            self.embedding = self.embedding_model or "gemini-embedding-001"
            try:
                from google import genai
            except Exception as exc:
                raise ValueError(
                    "Google embeddings require the google-genai package. "
                    "Install it with: pip install google-genai"
                ) from exc
            self.embedding_client_type = "google-genai"
            self.embedding_client = genai.Client(api_key=config["google_api_key"])
        elif self.embedding_provider == "none":
            self.embedding = None
            self.embedding_client = None
        else:
            # Default OpenAI setup (or OpenAI-compatible)
            self.embedding = self.embedding_model or "text-embedding-3-small"
            self.embedding_client = OpenAI(
                api_key=config.get("openai_api_key"),
                base_url=config.get("openai_api_base", "https://api.openai.com/v1"),
            )
        
        self.chroma_client = chromadb.Client(Settings(allow_reset=True))
        self.situation_collection = self.chroma_client.create_collection(name=name)

    def get_embedding(self, text):
        """Get embedding for a text using the appropriate API with error handling"""
        if self.embedding_provider == "none":
            print("🔄 Memory matching disabled (embedding_provider=none).")
            return None

        try:
            if self.embedding_provider == "google":
                if getattr(self, "embedding_client_type", "langchain") == "google-genai":
                    response = self.embedding_client.models.embed_content(
                        model=self.embedding,
                        contents=text,
                    )
                    if hasattr(response, "embeddings") and response.embeddings:
                        return response.embeddings[0].values
                    if hasattr(response, "embedding") and response.embedding:
                        return response.embedding.values
                    raise ValueError("Google embedding response missing embeddings.")
                return self.embedding_client.embed_query(text)
            response = self.embedding_client.embeddings.create(
                model=self.embedding, input=text
            )
            return response.data[0].embedding
        except Exception as e:
            # Handle quota errors, rate limits, and other API issues
            error_message = str(e).lower()
            if any(phrase in error_message for phrase in [
                "quota", "rate limit", "insufficient_quota", "billing", "429"
            ]):
                provider_label = "Google" if self.embedding_provider == "google" else "OpenAI"
                print(f"⚠️  {provider_label} embedding quota exceeded: {str(e)}")
                print("💡 Continuing without memory matching.")
            else:
                print(f"⚠️  Embedding error: {str(e)}")
                print("💡 Continuing without memory matching.")
            
            # Return None to indicate embedding failed
            return None

    def add_situations(self, situations_and_advice):
        """Add financial situations and their corresponding advice with error handling"""

        situations = []
        advice = []
        ids = []
        embeddings = []

        offset = self.situation_collection.count()

        for i, (situation, recommendation) in enumerate(situations_and_advice):
            embedding = self.get_embedding(situation)
            
            # Skip entries where embedding failed
            if embedding is None:
                print(f"⚠️  Skipping memory entry due to embedding failure: {situation[:50]}...")
                continue
                
            situations.append(situation)
            advice.append(recommendation)
            ids.append(str(offset + len(embeddings)))  # Use len(embeddings) for proper indexing
            embeddings.append(embedding)

        # Only add to collection if we have valid embeddings
        if embeddings:
            try:
                self.situation_collection.add(
                    documents=situations,
                    metadatas=[{"recommendation": rec} for rec in advice],
                    embeddings=embeddings,
                    ids=ids,
                )
                print(f"✅ Successfully added {len(embeddings)} memory entries")
            except Exception as e:
                print(f"⚠️  Failed to add memories to collection: {str(e)}")
        else:
            print("⚠️  No memories could be added due to embedding service issues")

    def get_memories(self, current_situation, n_matches=1):
        """Find matching recommendations using embeddings with graceful error handling"""
        query_embedding = self.get_embedding(current_situation)
        
        # If embedding failed, return empty results with explanation
        if query_embedding is None:
            print("🔄 Memory matching unavailable due to embedding service issues.")
            print("📈 Proceeding with analysis based on current data only.")
            return [
                {
                    "matched_situation": "Memory matching unavailable - continuing without historical context",
                    "recommendation": "Analyze current market conditions and fundamental data to make informed decisions",
                    "similarity_score": 0.0,
                }
            ]

        try:
            results = self.situation_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_matches,
                include=["metadatas", "documents", "distances"],
            )

            matched_results = []
            for i in range(len(results["documents"][0])):
                matched_results.append(
                    {
                        "matched_situation": results["documents"][0][i],
                        "recommendation": results["metadatas"][0][i]["recommendation"],
                        "similarity_score": 1 - results["distances"][0][i],
                    }
                )

            return matched_results
            
        except Exception as e:
            print(f"⚠️  Memory retrieval error: {str(e)}")
            print("📈 Proceeding with analysis based on current data only.")
            return [
                {
                    "matched_situation": "Memory retrieval failed - continuing without historical context",
                    "recommendation": "Focus on current fundamental analysis and market data for decision making",
                    "similarity_score": 0.0,
                }
            ]


if __name__ == "__main__":
    # Example usage
    matcher = FinancialSituationMemory()

    # Example data
    example_data = [
        (
            "High inflation rate with rising interest rates and declining consumer spending",
            "Consider defensive sectors like consumer staples and utilities. Review fixed-income portfolio duration.",
        ),
        (
            "Tech sector showing high volatility with increasing institutional selling pressure",
            "Reduce exposure to high-growth tech stocks. Look for value opportunities in established tech companies with strong cash flows.",
        ),
        (
            "Strong dollar affecting emerging markets with increasing forex volatility",
            "Hedge currency exposure in international positions. Consider reducing allocation to emerging market debt.",
        ),
        (
            "Market showing signs of sector rotation with rising yields",
            "Rebalance portfolio to maintain target allocations. Consider increasing exposure to sectors benefiting from higher rates.",
        ),
    ]

    # Add the example situations and recommendations
    matcher.add_situations(example_data)

    # Example query
    current_situation = """
    Market showing increased volatility in tech sector, with institutional investors 
    reducing positions and rising interest rates affecting growth stock valuations
    """

    try:
        recommendations = matcher.get_memories(current_situation, n_matches=2)

        for i, rec in enumerate(recommendations, 1):
            print(f"\nMatch {i}:")
            print(f"Similarity Score: {rec['similarity_score']:.2f}")
            print(f"Matched Situation: {rec['matched_situation']}")
            print(f"Recommendation: {rec['recommendation']}")

    except Exception as e:
        print(f"Error during recommendation: {str(e)}")
