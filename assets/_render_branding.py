"""Render the tts-bench logo as PNG outputs (social preview + favicon).

Run with: uv run --with Pillow python assets/_render_branding.py

Outputs:
  .github/social-preview.png   1280x640, for GitHub repo Social Preview
  _gh-pages/favicon.png         32x32, for gh-pages site
  _gh-pages/favicon-180.png    180x180, apple-touch-icon

Avoids an SVG renderer (cairo / playwright) by drawing the logo with
Pillow primitives — the mark is just 8 rounded rects + a monospaced
wordmark.
"""
from __future__ import annotations
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
GITHUB_DIR = REPO / ".github"
PAGES_DIR = REPO / "_gh-pages"

BG = (13, 13, 13)
FG_MARK = (0, 255, 136)
FG_TEXT = (242, 242, 242)

# Mark geometry: 8 bars, heights tracing a waveform peak.
# Each tuple is (y_offset_from_top_of_mark, height).
BAR_SHAPE = [
    (50, 20),
    (40, 40),
    (24, 72),
    (8, 104),
    (0, 120),
    (8, 104),
    (28, 64),
    (46, 28),
]
BAR_W = 10
BAR_GAP = 6  # x-step is BAR_W + BAR_GAP - BAR_W = BAR_GAP; effective stride = 16


def _load_mono(size: int) -> ImageFont.FreeTypeFont:
    """Find a bold monospace font that exists on Windows or Mac."""
    candidates = [
        "consolab.ttf",            # Windows Consolas Bold
        "C:/Windows/Fonts/consolab.ttf",
        "JetBrainsMono-Bold.ttf",
        "/Library/Fonts/JetBrainsMono-Bold.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "DejaVuSansMono-Bold.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_mark(draw: ImageDraw.ImageDraw, x: int, y: int, scale: float = 1.0, color=FG_MARK):
    """Draw the 8-bar waveform mark with top-left at (x, y)."""
    bar_w = max(1, round(BAR_W * scale))
    stride = round(16 * scale)
    radius = max(1, round(2 * scale))
    for i, (oy, oh) in enumerate(BAR_SHAPE):
        bx = x + i * stride
        by = y + round(oy * scale)
        bh = round(oh * scale)
        draw.rounded_rectangle([bx, by, bx + bar_w, by + bh], radius=radius, fill=color)


def render_social_preview(out: Path):
    """1280x640 with mark + wordmark centered horizontally, slightly above middle."""
    W, H = 1280, 640
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # Target ~60% of width, with generous padding so GitHub's avatar/text
    # overlays don't clip the logo. Iteratively size font to fit a budget.
    text = "tts-bench"
    safe_w = int(W * 0.72)  # max combined width
    font_size = 140
    while font_size > 60:
        font = _load_mono(font_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        scale = (font_size * 1.05) / 120  # mark height ~= cap height + a hair
        mark_total_w = round(((len(BAR_SHAPE) - 1) * 16 + BAR_W) * scale)
        gap = round(font_size * 0.3)
        total_w = mark_total_w + gap + text_w
        if total_w <= safe_w:
            break
        font_size -= 4
    else:
        font = _load_mono(font_size)
        bbox = draw.textbbox((0, 0), text, font=font)

    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    mark_h = round(120 * scale)

    start_x = (W - total_w) // 2
    cy = H // 2
    mark_y = cy - mark_h // 2
    _draw_mark(draw, start_x, mark_y, scale=scale)

    text_x = start_x + mark_total_w + gap
    text_y = cy - text_h // 2 - bbox[1]
    draw.text((text_x, text_y), text, font=font, fill=FG_TEXT)

    GITHUB_DIR.mkdir(exist_ok=True)
    img.save(out, "PNG", optimize=True)
    print(f"wrote {out.relative_to(REPO)} ({W}x{H})")


def render_favicon(out: Path, size: int, rounded: bool = True):
    """Square mark only, centered."""
    img = Image.new("RGB", (size, size), BG)
    draw = ImageDraw.Draw(img)

    if rounded and size >= 64:
        mask = Image.new("L", (size, size), 0)
        mdraw = ImageDraw.Draw(mask)
        mdraw.rounded_rectangle([0, 0, size, size], radius=size // 6, fill=255)
        # Re-create img with rounded corners via composite
        bg_img = Image.new("RGBA", (size, size), BG + (255,))
        clear = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        img = Image.composite(bg_img, clear, mask).convert("RGB")
        draw = ImageDraw.Draw(img)

    natural_w = (len(BAR_SHAPE) - 1) * 16 + BAR_W   # 122 wide
    natural_h = 120
    # Fit mark to ~62% of the icon
    scale = (size * 0.62) / natural_h
    mark_w = round(natural_w * scale)
    mark_h = round(natural_h * scale)
    x = (size - mark_w) // 2
    y = (size - mark_h) // 2
    _draw_mark(draw, x, y, scale=scale)

    img.save(out, "PNG", optimize=True)
    print(f"wrote {out.relative_to(REPO)} ({size}x{size})")


def main():
    render_social_preview(GITHUB_DIR / "social-preview.png")
    PAGES_DIR.mkdir(exist_ok=True)
    render_favicon(PAGES_DIR / "favicon.png", 32, rounded=False)
    render_favicon(PAGES_DIR / "favicon-180.png", 180, rounded=True)


if __name__ == "__main__":
    main()
