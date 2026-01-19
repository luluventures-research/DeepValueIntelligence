#!/usr/bin/env bash
###############################################################################
# make_slideshow_slides_fixed.sh — create a narrated slideshow video (TIMING FIXED)
#
# Dependencies
# • ffmpeg & ffprobe
# • whisper CLI
# • Python 3 with: pillow, pysrt, requests, numpy, scikit-learn, jieba,
#                  networkx, tqdm
# • Ollama running a local model (default gpt-oss:latest) on http://localhost:11434
#
# Usage:
#   ./make_slideshow_slides_fixed.sh BASENAME [CHUNK] [TITLE] [IMG_DIR] [LANG] [SPEED]
###############################################################################
set -euo pipefail

# ── CLI ---------------------------------------------------------------------
AUDIO_FILE_PATH=""
OUTPUT_DIR="."
SLIDES_PDF=""

# Collect positional arguments
POSITIONAL_ARGS=()

# Process all arguments, handling both named and positional
while [[ $# -gt 0 ]]; do
  case "$1" in
    --file)
      if [[ -n "$2" ]]; then
        AUDIO_FILE_PATH="$2"
        shift 2
      else
        echo "Error: --file requires an argument." >&2; exit 1
      fi
      ;;
    --output-dir)
      if [[ -n "$2" ]]; then
        OUTPUT_DIR="$2"
        shift 2
      else
        echo "Error: --output-dir requires an argument." >&2; exit 1
      fi
      ;;
    --slides)
      if [[ -n "$2" ]]; then
        SLIDES_PDF="$2"
        if [[ ! -f "$SLIDES_PDF" ]]; then
          echo "Error: PDF file not found: $SLIDES_PDF" >&2; exit 1
        fi
        shift 2
      else
        echo "Error: --slides requires a PDF file argument." >&2; exit 1
      fi
      ;;
    *)
      # Save positional arguments
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

# Restore positional arguments
set -- "${POSITIONAL_ARGS[@]}"

# If --file was used, derive BASE from it
if [[ -n "$AUDIO_FILE_PATH" ]]; then
  BASE=$(basename -- "$AUDIO_FILE_PATH")
  BASE="${BASE%.*}"
else
  # If --file is not used, the first positional argument is BASE
  if [[ -z "${1:-}" ]]; then
    echo "Error: Missing BASE name or --file argument." >&2
    exit 1
  fi
  BASE="$1"
  shift # Consume BASE
fi

# Now process remaining positional arguments
# Track if CHUNK was explicitly provided
if [[ -n "${1:-}" ]]; then
  CHUNK="$1"
  CHUNK_EXPLICIT=true
else
  CHUNK=30
  CHUNK_EXPLICIT=false
fi
TITLE="${2:-$(echo "$BASE" | tr '_-' ' ')}"
IMG_DIR="${3:-$BASE}"
LANG="${4:-English}"
SPEED="${5:-1.05}"

# ── Layout ------------------------------------------------------------------
W=1920 ; H=1080
TITLE_SIZE=96
BODY_SIZE_EN=45                   # English bullets
BODY_SIZE_ZH=55                   # Chinese bullets
LEFT_PAD=220 ; RIGHT_PAD=80
LINE_SP=34  ; TOP_MARGIN=100
BG="#1A2B3C" ; FG="#FFFFFF"

# ── Video timing settings --------------------------------------------------
TARGET_FPS=2                     # Fixed FPS for predictable timing

# ── Ollama ------------------------------------------------------------------
: "${OLLAMA_URL:=http://localhost:11434/api/generate}"
: "${OLLAMA_MODEL:=gpt-oss:latest}"

LANG_LC=$(echo "$LANG" | tr '[:upper:]' '[:lower:]')
IS_ZH=$([[ "$LANG_LC" =~ ^(chinese|zh|zh_cn|zh-cn)$ ]] && echo 1 || echo 0)
BODY_SIZE=$([ "$IS_ZH" -eq 1 ] && echo "$BODY_SIZE_ZH" || echo "$BODY_SIZE_EN")

# ── Paths -------------------------------------------------------------------
# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

if [[ -n "$AUDIO_FILE_PATH" ]]; then
  AUDIO_INPUT="$AUDIO_FILE_PATH"
elif [[ -f "$BASE.wav" ]]; then AUDIO_INPUT="$BASE.wav"
elif [[ -f "$BASE.m4a" ]]; then AUDIO_INPUT="$BASE.m4a"
else echo "❌  Neither $BASE.wav, $BASE.m4a, nor a file specified with --file found" >&2 ; exit 1 ; fi

POD="$OUTPUT_DIR/$BASE.podcast.wav"
SRT_FILE="$OUTPUT_DIR/${BASE}.podcast.srt"
SLIDES_DIR="$OUTPUT_DIR/${BASE}_slides"
OUT_MP4="$OUTPUT_DIR/${BASE}_slides.mp4"

# Clean previous artifacts
if [[ -d "$SLIDES_DIR" ]]; then
  echo "🧹 Cleaning previous slides folder: $SLIDES_DIR"
  rm -rf "$SLIDES_DIR"
fi
rm -f "$POD" "$SRT_FILE" "$OUT_MP4"

# ── 1. Speed-up audio -------------------------------------------------------
ORIGINAL_DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$AUDIO_INPUT")
echo "🎵 Original audio duration: ${ORIGINAL_DURATION}s"
echo "⚡ Applying speed multiplier: ${SPEED}x"

ffmpeg -y -i "$AUDIO_INPUT" -filter:a atempo="$SPEED" \
       -ac 2 -ar 44100 -sample_fmt s16 "$POD"

# Get audio duration
AUDIO_DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$POD")
echo "✅ Sped-up audio duration: ${AUDIO_DURATION}s"

# Fresh slides dir
mkdir -p "$SLIDES_DIR"

# ── 2. Handle PDF slides or transcription ----------------------------------
if [[ -n "$SLIDES_PDF" ]]; then
  echo "🎯 Using PDF slides from: $SLIDES_PDF"

  # Extract PDF pages to PNG images using Python (PyMuPDF/fitz)
  echo "📄 Extracting PDF pages..."

  export SLIDES_PDF SLIDES_DIR

  python <<'PDFPY'
import fitz  # PyMuPDF
import os
from pathlib import Path
from PIL import Image

pdf_path = os.environ["SLIDES_PDF"]
slides_dir = Path(os.environ["SLIDES_DIR"])

# Open PDF
pdf_doc = fitz.open(pdf_path)
page_count = len(pdf_doc)

print(f"📄 Extracting {page_count} pages from PDF...")

# Extract each page as PNG and remove NotebookLM watermark
for page_num in range(page_count):
    page = pdf_doc[page_num]
    # Render page at 150 DPI (zoom factor = 150/72 = 2.08)
    zoom = 2.08
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)

    # Save to temporary path
    temp_path = slides_dir / f"temp_page-{page_num + 1}.png"
    pix.save(temp_path)

    # Load with PIL to remove watermark and add custom watermark
    img = Image.open(temp_path)
    width, height = img.size

    # Crop out bottom-right watermark area
    # NotebookLM watermark is typically in bottom-right corner
    # Optimized dimensions for precise watermark removal
    crop_bottom = 60   # pixels to remove from bottom
    crop_right = 230   # pixels to remove from right

    # Create a white rectangle to cover the original watermark area
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    # Draw white rectangle over watermark area (bottom-right corner)
    watermark_box = [width - crop_right, height - crop_bottom, width, height]
    draw.rectangle(watermark_box, fill='white')

    # Add custom watermark "luluvc.com"
    watermark_text = "luluvc.com"

    # Try to use a nice font, fall back to default if not available
    try:
        # Try different font paths for macOS
        font_size = 28
        font_paths = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/HelveticaNeue.ttc",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]
        font = None
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    # Get text bounding box to position it properly
    bbox = draw.textbbox((0, 0), watermark_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Position text in bottom-right corner with padding
    padding_right = 20
    padding_bottom = 15
    text_x = width - text_width - padding_right
    text_y = height - text_height - padding_bottom

    # Draw the text in a subtle gray color
    text_color = (128, 128, 128)  # Gray color
    draw.text((text_x, text_y), watermark_text, fill=text_color, font=font)

    # Save the cleaned image with new watermark
    output_path = slides_dir / f"pdf_page-{page_num + 1}.png"
    img.save(output_path)

    # Remove temp file
    temp_path.unlink()

pdf_doc.close()
print(f"✅ Extracted {page_count} pages, removed NotebookLM watermarks, and added luluvc.com branding")
PDFPY

  # Count number of slides
  PDF_SLIDE_COUNT=$(ls -1 "$SLIDES_DIR"/pdf_page-*.png 2>/dev/null | wc -l)

  if [[ $PDF_SLIDE_COUNT -eq 0 ]]; then
    echo "❌ Failed to extract PDF slides" >&2
    exit 1
  fi

  echo "✅ Extracted $PDF_SLIDE_COUNT slides from PDF"

  # Auto-calculate CHUNK if not explicitly provided
  if [[ "$CHUNK_EXPLICIT" == "false" ]]; then
    # Calculate CHUNK as total audio duration divided by number of slides
    CHUNK=$(echo "scale=2; $AUDIO_DURATION / $PDF_SLIDE_COUNT" | bc)
    echo "🔢 Auto-calculated CHUNK: ${CHUNK}s per slide (${AUDIO_DURATION}s ÷ ${PDF_SLIDE_COUNT} slides)"
  fi

else
  # ── 2. Transcribe ---------------------------------------------------------
  whisper "$POD" --model small --language "$LANG" --device cpu \
          --fp16 False --task transcribe --output_format srt -o "$OUTPUT_DIR"

  # Rename whisper's output to our standard name
  mv "${POD%.*}.srt" "$SRT_FILE"
fi

# ── 3. Generate PNG frames --------------------------------------------------
export W H TITLE_SIZE BODY_SIZE LEFT_PAD RIGHT_PAD LINE_SP TOP_MARGIN \
       BG FG TITLE SLIDES_DIR IMG_DIR IS_ZH SRT_FILE CHUNK \
       OLLAMA_URL OLLAMA_MODEL TARGET_FPS SLIDES_PDF AUDIO_DURATION

python <<'PY'
import os, re, json, requests, pysrt, numpy as np, tqdm
import jieba, jieba.analyse, networkx as nx
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

# ── env ---------------------------------------------------------------------
W,H        = int(os.environ["W"]), int(os.environ["H"])
TS         = int(os.environ["TITLE_SIZE"])
BS         = int(os.environ["BODY_SIZE"])
LEFT,RIGHT = int(os.environ["LEFT_PAD"]), int(os.environ["RIGHT_PAD"])
SP,TOP     = int(os.environ["LINE_SP"]), int(os.environ["TOP_MARGIN"])
BG,FG      = os.environ["BG"], os.environ["FG"]
TITLE      = os.environ["TITLE"]
SLIDES     = Path(os.environ["SLIDES_DIR"]); IMG_DIR=Path(os.environ["IMG_DIR"])
IS_ZH      = bool(int(os.environ["IS_ZH"]))
SRT_FILE   = os.environ.get("SRT_FILE", ""); CHUNK=float(os.environ["CHUNK"])
OLLAMA_URL = os.environ["OLLAMA_URL"]; OLLAMA_MODEL=os.environ["OLLAMA_MODEL"]
TARGET_FPS = int(os.environ["TARGET_FPS"])
SLIDES_PDF = os.environ.get("SLIDES_PDF", "")
AUDIO_DURATION = float(os.environ.get("AUDIO_DURATION", "0"))

# ── PDF Slides Mode ---------------------------------------------------------
if SLIDES_PDF:
    print("🎬 PDF Slides Mode: Processing slides from PDF")

    # Find all extracted PDF slide images
    pdf_slides = sorted(SLIDES.glob("pdf_page-*.png"), key=lambda p: int(re.search(r'-(\d+)\.png$', p.name).group(1)))

    if not pdf_slides:
        print("❌ No PDF slides found!")
        exit(1)

    num_slides = len(pdf_slides)

    # Calculate duration for each slide
    # Each slide gets CHUNK seconds, except the last one which gets the remaining time
    slide_durations = []
    for i in range(num_slides - 1):
        slide_durations.append(CHUNK)

    # Last slide gets remaining time
    time_used = CHUNK * (num_slides - 1)
    last_slide_duration = max(CHUNK, AUDIO_DURATION - time_used)
    slide_durations.append(last_slide_duration)

    print(f"📊 Total slides: {num_slides}")
    print(f"⏱️  Audio duration: {AUDIO_DURATION:.2f}s")
    print(f"⏱️  Duration per slide: {CHUNK}s")
    print(f"⏱️  Last slide duration: {last_slide_duration:.2f}s")

    # Frame counter
    frame = 0

    def save_frames(img, duration_sec):
        """Save frames for specified duration at TARGET_FPS"""
        global frame
        frames_needed = max(1, int(duration_sec * TARGET_FPS))
        for _ in range(frames_needed):
            img.save(SLIDES/f"frame_{frame:06d}.png")
            frame += 1

    # Process each PDF slide
    for i, pdf_slide_path in enumerate(tqdm.tqdm(pdf_slides, desc="Processing PDF slides")):
        # Load the PDF slide image
        slide_img = Image.open(pdf_slide_path).convert("RGB")

        # Resize to fit within the frame while maintaining aspect ratio
        slide_img.thumbnail((W, H), Image.Resampling.LANCZOS)

        # Create a background and center the slide
        bg = Image.new("RGB", (W, H), BG)
        x_offset = (W - slide_img.width) // 2
        y_offset = (H - slide_img.height) // 2
        bg.paste(slide_img, (x_offset, y_offset))

        # Save frames for this slide with its specific duration
        save_frames(bg, slide_durations[i])

    # Write frame count
    (SLIDES/"frames_count.txt").write_text(str(frame))
    print(f"✅ Generated {frame} frames from {num_slides} PDF slides")

    # Exit early - skip all the transcription-based slide generation
    exit(0)

# ── 3-a. Slice transcript ---------------------------------------------------
subs=pysrt.open(SRT_FILE)
blocks={}
for s in subs:
    idx=s.start.ordinal//1000//CHUNK
    blocks.setdefault(idx,[]).append(s.text.replace('\n',' '))

# ── 3-b. Summarisation helpers ---------------------------------------------
FIL_ZH=re.compile(r'(嗯+|啊+|呃+|哼+|呐+|呗+|啰+|哦+|噢+|吗+|然后|就是|那个|这个|其实|那么)')
FIL_EN_L=re.compile(r'^(and|but|so|because|well|okay|alright|anyway)\s+',re.I)
FIL_EN_I=re.compile(r'\b(?:so|well|really|basically|literally|just|you know|uh|um|uh-huh|like|sort of|kind of|kinda)\b',re.I)

def call_ollama(prompt:str)->str:
    try:
        r=requests.post(OLLAMA_URL,json={"model":OLLAMA_MODEL,"prompt":prompt,"stream":False},timeout=60)
        r.raise_for_status()
        return r.json().get("response","").strip()
    except Exception as e:
        print("⚠️  Ollama call failed:",e); return ""

def en_bullets(txt:str):
    resp=call_ollama("Give me 3-4 concise professional bullet points (≤15 words each) that summarise:\n\n"+txt)
    parts=re.split(r'(?:^\s*[\-\*•]\s+|\n)',resp,flags=re.M) if resp else []
    if len(parts)<2: parts=re.split(r'(?<=[.!?])\s+',resp)
    out=[]
    for p in parts:
        p=FIL_EN_I.sub("",p); p=FIL_EN_L.sub("",p).strip().capitalize()
        p=re.sub(r'\s-\s',' – ',p)        # English: normalize " - " → en dash with spaces
        if len(p.split())>=3 and p not in out:
            out.append(p if p.endswith(('.','!','?')) else p+'.')
        if len(out)==4: break
    return out

def _normalize_zh_dashes(s:str)->str:
    # Case 1: ranges like 2024-2025 or A-B → en dash without spaces
    s = re.sub(r'(?<=\w)\s*[-−–—]\s*(?=\w)', '–', s)
    # Case 2: any remaining standalone dash-like → en dash with single spaces
    s = re.sub(r'\s*[-−–—]\s*', ' – ', s)
    # Tidy spacing
    s = re.sub(r'\s{2,}', ' ', s).strip()
    return s

def zh_bullets(txt:str):
    resp=call_ollama("请用专业简体中文将下列内容概括成 3-4 个要点，每点 ≤25 字：\n\n"+txt)
    out=[s.strip("•- \n") for s in re.split(r'[；\n]',resp) if s.strip()] if resp else []
    out=[_normalize_zh_dashes(FIL_ZH.sub("",s)) for s in out]
    return out[:4]

def textrank_cn(txt,k=4):
    txt=FIL_ZH.sub("",txt)
    sents=[s for s in re.split(r'[。！？]',txt) if s.strip()]
    if not sents: return []
    vectorizer = TfidfVectorizer()
    transformed = vectorizer.fit_transform(sents)
    similarity_matrix = transformed * transformed.T
    M = similarity_matrix.toarray()
    np.fill_diagonal(M,0); pr=nx.pagerank(nx.from_numpy_array(M))
    pick=[sents[i] for i in sorted(pr,key=pr.get,reverse=True)[:k]]
    pick=[s if s.endswith("。") else s+"。" for s in pick]
    return [_normalize_zh_dashes(s) for s in pick]

def bullets(txt:str):
    return (zh_bullets(txt) or textrank_cn(txt)) if IS_ZH else en_bullets(txt)

# ── 3-c. Font helper (language-aware; probes en dash too) -------------------
def font_ok(sz,bold=False):
    if IS_ZH:
        order=[
            "/Library/Fonts/NotoSansSC-Regular.otf",
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/HelveticaNeue.ttc",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]
    else:
        order=[
            "/System/Library/Fonts/HelveticaNeue.ttc",
            "/System/Library/Fonts/SFNS.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/NotoSansSC-Regular.otf",
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
        ]
        if bold:
            order.insert(2,"/System/Library/Fonts/Supplemental/Arial Bold.ttf")
    for p in order:
        try:
            f=ImageFont.truetype(p,sz)
            probe = "测——–-" if IS_ZH else "A-–—"
            if ImageDraw.Draw(Image.new("RGB",(1,1))).textbbox((0,0),probe,font=f):
                return f
        except Exception:
            continue
    return ImageFont.load_default()

ft_title,ft_body=font_ok(TS,True),font_ok(BS)
tmp=ImageDraw.Draw(Image.new("RGB",(1,1)))
txtw=lambda t,f: tmp.textbbox((0,0),t,font=f)[2]
WRAP_MAX=W-LEFT-RIGHT-40
def wrap(t):
    toks=list(t) if IS_ZH else t.split()
    buf=""; lines=[]
    for tok in toks:
        trial=(buf+tok) if IS_ZH else f"{buf} {tok}".strip()
        if txtw(trial,ft_body)<=WRAP_MAX: buf=trial
        else: lines.append(buf); buf=tok
    if buf: lines.append(buf)
    return lines

# ── 3-d. Images (cycled / round-robin) -------------------------------------
def natural_key(p): return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)',p.stem)]
pics=sorted([p for p in Path(os.environ["IMG_DIR"]).glob('*') if p.suffix.lower() in {'.png','.jpg','.jpeg'}],key=natural_key)

# Optional splash shown once if present
splash=next((p for p in pics if re.search(r'(thumbnail|封面)',p.stem,re.I)),None)
if splash: pics.remove(splash)

have_pics = len(pics) > 0
img_idx = 0

# ── 3-e. Save helper with frame duplication for timing ---------------------
frame=0
def save_frames(img, duration_sec):
    """Save frames for specified duration at TARGET_FPS"""
    global frame
    frames_needed = max(1, int(duration_sec * TARGET_FPS))
    for _ in range(frames_needed):
        img.save(SLIDES/f"frame_{frame:06d}.png")
        frame += 1

# splash frame (show for 2 seconds) -----------------------------------------
if splash:
    sp=Image.open(splash).convert("RGB"); sp.thumbnail((W,H))
    bg=Image.new("RGB",(W,H),BG); bg.paste(sp,((W-sp.width)//2,(H-sp.height)//2))
    save_frames(bg, 2.0)

# ── 3-f. Build slides with proper timing -----------------------------------
total_chunks = len(blocks)
for idx in tqdm.tqdm(sorted(blocks)):
    text=" ".join(blocks[idx])[:2000]
    
    # Calculate timing: each chunk gets equal share of remaining time
    if have_pics:
        # With images: split chunk time between bullet slide and image slide
        bullet_duration = CHUNK * 0.6  # 60% for bullets
        image_duration = CHUNK * 0.4   # 40% for images
    else:
        # Without images: full chunk time for bullet slide
        bullet_duration = CHUNK

    # bullet slide
    slide=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(slide)
    d.text(((W-txtw(TITLE,ft_title))//2,TOP),TITLE,font=ft_title,fill=FG)
    y=TOP+TS+70
    for b in bullets(text):
        first=True
        for ln in wrap(b):
            prefix=("• " if not IS_ZH else "•") if first else "  "
            d.text((LEFT,y),prefix+ln,font=ft_body,fill=FG)
            y+=BS+SP; first=False
        y+=SP
        if y>H-120: break
    save_frames(slide, bullet_duration)

    # image slide — round-robin (only if images exist)
    if have_pics:
        p = pics[img_idx]; img_idx = (img_idx + 1) % len(pics)
        im=Image.open(p).convert("RGB"); im.thumbnail((W,H))
        bg=Image.new("RGB",(W,H),BG); bg.paste(im,((W-im.width)//2,(H-im.height)//2))
        save_frames(bg, image_duration)

(SLIDES/"frames_count.txt").write_text(str(frame))
PY

# ── 4. Assemble video with fixed FPS ---------------------------------------
FRAMES=$(<"$SLIDES_DIR/frames_count.txt")

# Use fixed FPS - frames are already generated for correct timing
ffmpeg -y -framerate "$TARGET_FPS" -pattern_type glob -i "$SLIDES_DIR/frame_*.png" \
       -i "$POD" -c:v libx264 -r 30 -pix_fmt yuv420p -preset veryfast -crf 18 \
       -c:a aac -b:a 192k -shortest "$OUT_MP4"

echo "✅  $OUT_MP4 created — timing fixed for images; frames properly synchronized with audio"
###############################################################################