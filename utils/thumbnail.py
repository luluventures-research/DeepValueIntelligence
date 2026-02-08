import os
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import requests


def _slugify_name(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", (name or "").strip())
    slug = slug.strip("_")
    return slug or "company"


def _load_font(size: int, bold: bool = False, cjk: bool = False):
    from PIL import ImageFont

    if cjk:
        candidates = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
        ]
    elif bold:
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial Bold.ttf",
        ]
    else:
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf",
        ]

    for font_path in candidates:
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _fetch_logo_url(ticker: str):
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key or not ticker:
        return None, None, None

    try:
        resp = requests.get(
            "https://finnhub.io/api/v1/stock/profile2",
            params={"symbol": ticker.upper(), "token": api_key},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json() or {}
        logo_url = data.get("logo")
        weburl = data.get("weburl")
        company_name = data.get("name")
        if isinstance(logo_url, str) and logo_url.startswith("http"):
            return logo_url, weburl, company_name
        return None, weburl, company_name
    except Exception:
        return None, None, None
    return None, None, None


def _fetch_logo_url_from_clearbit(weburl: Optional[str]) -> Optional[str]:
    if not weburl:
        return None
    try:
        domain = re.sub(r"^https?://", "", weburl).split("/")[0].strip()
        if not domain:
            return None
        return f"https://logo.clearbit.com/{domain}"
    except Exception:
        return None


def _fetch_logo_image(logo_url: Optional[str]):
    if not logo_url:
        return None
    try:
        from PIL import Image
    except Exception:
        return None

    try:
        resp = requests.get(logo_url, timeout=10)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGBA")
    except Exception:
        return None


def _extract_image_from_response(response):
    try:
        if hasattr(response, "parts") and response.parts:
            parts = response.parts
        elif hasattr(response, "candidates") and response.candidates:
            parts = []
            for cand in response.candidates:
                content = getattr(cand, "content", None)
                if content and getattr(content, "parts", None):
                    parts.extend(content.parts)
        else:
            parts = []
    except Exception:
        parts = []

    try:
        from PIL import Image
    except Exception:
        return None

    for part in parts:
        if hasattr(part, "inline_data") and getattr(part.inline_data, "data", None):
            try:
                return Image.open(BytesIO(part.inline_data.data))
            except Exception:
                continue
        if hasattr(part, "as_image"):
            try:
                img = part.as_image()
                if hasattr(img, "size"):
                    return img
            except Exception:
                continue
    return None


def _generate_background(client, types, prompt: str, model: str):
    fallbacks = [model, "gemini-2.5-flash-image"]
    if model == "gemini-2.5-flash-image":
        fallbacks = [model]

    last_error = None
    for model_name in fallbacks:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(aspect_ratio="16:9", image_size="2K"),
                ),
            )
            image = _extract_image_from_response(response)
            if image is not None:
                return image, None
        except Exception as exc:
            last_error = exc
            continue
    return None, last_error


def _wrap_text(draw, text: str, font, max_width: int) -> List[str]:
    words = text.split(" ")
    lines: List[str] = []
    cur = ""
    for word in words:
        candidate = word if not cur else f"{cur} {word}"
        width = draw.textbbox((0, 0), candidate, font=font)[2]
        if width <= max_width or not cur:
            cur = candidate
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def _draw_overlay(
    image,
    ticker: str,
    title: str,
    subtitle: str,
    date_line: str,
    logo,
    cjk: bool = False,
):
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

    canvas = image.convert("RGB").resize((1920, 1080), Image.Resampling.LANCZOS)
    canvas = ImageEnhance.Contrast(canvas).enhance(1.18)
    canvas = ImageEnhance.Color(canvas).enhance(1.2)
    canvas = canvas.convert("RGBA")
    w, h = canvas.size

    # Left readability gradient.
    mask = Image.new("L", (w, h), 0)
    mdraw = ImageDraw.Draw(mask)
    for x in range(w):
        a = int(235 * (1 - x / w))
        mdraw.line([(x, 0), (x, h)], fill=max(0, a))
    layer = Image.new("RGBA", (w, h), (6, 12, 24, 0))
    layer.putalpha(mask)
    canvas = Image.alpha_composite(canvas, layer)

    # Bottom vignette.
    vignette = Image.new("L", (w, h), 0)
    vdraw = ImageDraw.Draw(vignette)
    vdraw.ellipse((-500, 320, w + 500, h + 800), fill=230)
    vignette = vignette.filter(ImageFilter.GaussianBlur(120))
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    shadow.putalpha(vignette)
    canvas = Image.alpha_composite(canvas, shadow)

    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(96, bold=True, cjk=cjk)
    subtitle_font = _load_font(56, bold=True, cjk=cjk)
    date_font = _load_font(40, bold=False, cjk=False)
    badge_font = _load_font(32, bold=True, cjk=False)
    watermark_font = _load_font(260, bold=True, cjk=False)

    # Badge.
    draw.rounded_rectangle((88, 92, 430, 164), radius=18, fill=(242, 82, 45, 238))
    draw.text((114, 113), "DEEP VALUE", fill=(255, 255, 255, 255), font=badge_font)

    # Watermark ticker.
    draw.text((880, 735), ticker[:8], fill=(255, 255, 255, 32), font=watermark_font)

    # Title/subtitle/date.
    x = 150
    title_start_y = 250
    text_top = title_start_y
    title_lines = _wrap_text(draw, title, title_font, 1020)[:2]
    title_line_height = draw.textbbox((0, 0), "Ag", font=title_font)[3]
    title_step = max(114, title_line_height + 18)
    subtitle_height = draw.textbbox((0, 0), subtitle, font=subtitle_font)[3]

    title_widths = [draw.textbbox((0, 0), line, font=title_font)[2] for line in title_lines]
    max_title_width = max(title_widths) if title_widths else 0
    subtitle_width = draw.textbbox((0, 0), subtitle, font=subtitle_font)[2]
    subtitle_y = title_start_y + (len(title_lines) * title_step) + 14
    date_box_y = subtitle_y + max(100, subtitle_height + 34)

    date_text_bbox = draw.textbbox((0, 0), date_line, font=date_font)
    date_w = date_text_bbox[2] - date_text_bbox[0]
    date_h = date_text_bbox[3] - date_text_bbox[1]
    date_pad_x = 22
    date_pad_y = 10
    date_box_w = date_w + date_pad_x * 2
    date_box_h = date_h + date_pad_y * 2

    # Backgrounds: keep them optional to avoid covering the generated image.
    text_panel_alpha = 0
    date_pill_alpha = 0

    panel_padding_x = 36
    panel_padding_y = 28
    text_right = x + max(max_title_width, subtitle_width, date_box_w)
    text_bottom = date_box_y + date_box_h
    panel_left = max(0, x - panel_padding_x)
    panel_top = max(0, text_top - panel_padding_y)
    panel_right = min(w, text_right + panel_padding_x)
    panel_bottom = min(h, text_bottom + panel_padding_y)

    if text_panel_alpha > 0:
        panel_w = max(1, int(panel_right - panel_left))
        panel_h = max(1, int(panel_bottom - panel_top))
        panel = Image.new("RGBA", (panel_w, panel_h), (12, 22, 40, text_panel_alpha))
        panel_mask = Image.new("L", (panel_w, panel_h), 0)
        pdraw = ImageDraw.Draw(panel_mask)
        pdraw.rounded_rectangle((0, 0, panel_w, panel_h), radius=28, fill=255)
        canvas.paste(panel, (int(panel_left), int(panel_top)), panel_mask)

    y = title_start_y
    for line in title_lines:
        draw.text((x, y), line, font=title_font, fill=(255, 255, 255, 255), stroke_width=4, stroke_fill=(0, 0, 0, 170))
        y += title_step

    draw.text((x, subtitle_y), subtitle, font=subtitle_font, fill=(235, 242, 252, 255), stroke_width=3, stroke_fill=(0, 0, 0, 150))

    if date_pill_alpha > 0:
        draw.rounded_rectangle(
            (x, date_box_y, x + date_box_w, date_box_y + date_box_h),
            radius=16,
            fill=(24, 46, 82, date_pill_alpha),
        )
    draw.text((x + date_pad_x, date_box_y + date_pad_y), date_line, font=date_font, fill=(223, 234, 248, 255))

    # Logo card.
    if logo is not None:
        logo_img = logo.copy()
        logo_img.thumbnail((206, 206), Image.Resampling.LANCZOS)
        lw, lh = logo_img.size
        cx, cy = w - lw - 130, 104
        card = Image.new("RGBA", (lw + 52, lh + 52), (255, 255, 255, 245))
        card_mask = Image.new("L", (lw + 52, lh + 52), 0)
        cdraw = ImageDraw.Draw(card_mask)
        cdraw.rounded_rectangle((0, 0, lw + 52, lh + 52), radius=24, fill=255)
        canvas.paste(card, (cx - 26, cy - 26), card_mask)
        canvas.paste(logo_img, (cx, cy), logo_img)
    else:
        fallback_font = _load_font(52, bold=True, cjk=False)
        draw.rounded_rectangle((w - 360, 96, w - 88, 196), radius=18, fill=(18, 34, 60, 232))
        draw.text((w - 330, 124), ticker[:8], fill=(255, 255, 255, 255), font=fallback_font)

    return canvas.convert("RGB")


def generate_thumbnails(
    company_name: str,
    report_dir: Path,
    api_key: Optional[str] = None,
    model: str = "gemini-3-pro-image-preview",
    ticker: Optional[str] = None,
    analysis_date: Optional[str] = None,
) -> List[Path]:
    try:
        from google import genai
        from google.genai import types
    except Exception as exc:
        print(f"Thumbnail generation unavailable (google-genai): {exc}")
        return []

    report_dir.mkdir(parents=True, exist_ok=True)
    images_dir = report_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key) if api_key else genai.Client()

    symbol = (ticker or company_name or "COMPANY").strip().upper()
    safe_company = _slugify_name(company_name or symbol)

    # Remove older generated versions for this company.
    for old in images_dir.glob(f"{safe_company}_thumbnail_*.png"):
        try:
            old.unlink()
        except Exception:
            pass

    today = datetime.now().strftime("%Y-%m-%d")
    if analysis_date and analysis_date != today:
        date_text = f"Updated {today} | Data {analysis_date}"
    else:
        date_text = f"Updated {today}"

    prompts: Dict[str, str] = {
        "en": (
            f"Create a striking premium stock-research thumbnail background, 16:9, {symbol} Deep Value Analysis. "
            "Professional, cinematic, high contrast, modern finance mood. No text, no logos, no watermark."
        ),
        "cn": (
            f"创建一张高端吸睛的股票研究缩略图背景，16:9，{symbol} 深度价值分析。"
            "整体专业、电影感、高对比度、现代金融氛围。不要文字、不要logo、不要水印。"
        ),
    }

    logo_url, weburl, profile_company_name = _fetch_logo_url(symbol)
    logo = _fetch_logo_image(logo_url)
    if logo is None:
        clearbit_url = _fetch_logo_url_from_clearbit(weburl)
        logo = _fetch_logo_image(clearbit_url)

    display_name = (company_name or "").strip()
    if not display_name or display_name.upper() == symbol:
        if isinstance(profile_company_name, str) and profile_company_name.strip():
            display_name = profile_company_name.strip()
    if not display_name:
        display_name = symbol

    outputs: List[Path] = []
    for suffix, prompt in prompts.items():
        background, err = _generate_background(client, types, prompt, model)
        if background is None:
            print(f"Thumbnail background generation failed for {suffix}: {err}")
            continue

        if suffix == "cn":
            title = f"{display_name} 深度价值分析"
            subtitle = "估值，增长，风险评估"
            cjk = True
        else:
            title = f"{display_name} Deep Value Analysis"
            subtitle = "Valuation, Growth, and Risks"
            cjk = False

        final_image = _draw_overlay(
            image=background,
            ticker=symbol,
            title=title,
            subtitle=subtitle,
            date_line=date_text,
            logo=logo,
            cjk=cjk,
        )

        out = images_dir / f"{safe_company}_thumbnail_{suffix}.png"
        final_image.save(out)
        outputs.append(out)

    return outputs
