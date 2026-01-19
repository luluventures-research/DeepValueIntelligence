import chromadb
from chromadb.config import Settings
from openai import OpenAI


class FinancialSituationMemory:
    def __init__(self, name, config):
        self.config = config
        
        # Determine embedding model and API endpoint based on configuration
        if config["backend_url"] == "http://localhost:11434/v1":
            # Ollama local setup
            self.embedding = "nomic-embed-text"
            self.embedding_client = OpenAI(base_url=config["backend_url"])
        else:
            # Check if we're using Google models for the main LLMs
            deep_model = config.get("deep_think_llm", "")
            quick_model = config.get("quick_think_llm", "")
            
            using_gemini = (
                deep_model.startswith(("gemini", "google")) or 
                quick_model.startswith(("gemini", "google"))
            )
            
            if using_gemini:
                # When using Gemini models, use OpenAI API for embeddings
                if not config.get("openai_api_key"):
                    raise ValueError(
                        "When using Gemini models, OPENAI_API_KEY is required for embedding functionality. "
                        "Please set both GOOGLE_API_KEY (for Gemini) and OPENAI_API_KEY (for embeddings)."
                    )
                self.embedding = "text-embedding-3-small"
                self.embedding_client = OpenAI(
                    api_key=config.get("openai_api_key"),
                    base_url=config.get("openai_api_base", "https://api.openai.com/v1")
                )
            else:
                # Standard OpenAI setup
                self.embedding = "text-embedding-3-small"
                self.embedding_client = OpenAI(base_url=config["backend_url"])
        
        self.chroma_client = chromadb.Client(Settings(allow_reset=True))
        self.situation_collection = self.chroma_client.create_collection(name=name)

    def get_embedding(self, text):
        """Get embedding for a text using the appropriate API with error handling"""
        try:
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
                print(f"⚠️  OpenAI embedding quota exceeded: {str(e)}")
                print("💡 Continuing without memory matching. Consider checking your OpenAI billing at https://platform.openai.com/usage")
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
