#!/usr/bin/env python3
"""Generate Deep Value Intelligence logo in the style of the original Lulu Research logo."""

from PIL import Image, ImageDraw, ImageFont
import os

# Image dimensions (same as original)
WIDTH = 2400
HEIGHT = 500

# Colors
PURPLE = (139, 127, 199)  # #8B7FC7 - the purple/violet color
WHITE = (255, 255, 255)
LIGHT_PURPLE = (200, 195, 225)  # For circle outlines

# Grid settings - smaller to match proportions
GRID_SIZE = 8
CIRCLE_RADIUS = 16
CIRCLE_SPACING = 42
GRID_START_X = 50
GRID_START_Y = 80

# Create image with transparent background
img = Image.new('RGBA', (WIDTH, HEIGHT), (255, 255, 255, 0))
draw = ImageDraw.Draw(img)

# Define "D" pattern - which cells should be filled
def is_d_pattern(row, col):
    """Return True if this cell should be filled to form a 'D' shape."""
    # Left vertical bar (column 0)
    if col == 0:
        return True
    # Top horizontal (row 0, columns 1-5)
    if row == 0 and 1 <= col <= 5:
        return True
    # Bottom horizontal (row 7, columns 1-5)
    if row == GRID_SIZE - 1 and 1 <= col <= 5:
        return True
    # Right curve - top right corner
    if row == 1 and col in [5, 6]:
        return True
    # Right curve - upper middle
    if row == 2 and col in [6, 7]:
        return True
    # Right curve - middle
    if row in [3, 4] and col == 7:
        return True
    # Right curve - lower middle
    if row == 5 and col in [6, 7]:
        return True
    # Right curve - bottom right corner
    if row == 6 and col in [5, 6]:
        return True
    return False

# Draw circle grid
for row in range(GRID_SIZE):
    for col in range(GRID_SIZE):
        x = GRID_START_X + col * CIRCLE_SPACING
        y = GRID_START_Y + row * CIRCLE_SPACING

        if is_d_pattern(row, col):
            # Filled circle
            draw.ellipse(
                [x - CIRCLE_RADIUS, y - CIRCLE_RADIUS,
                 x + CIRCLE_RADIUS, y + CIRCLE_RADIUS],
                fill=PURPLE
            )
        else:
            # Outline circle
            draw.ellipse(
                [x - CIRCLE_RADIUS, y - CIRCLE_RADIUS,
                 x + CIRCLE_RADIUS, y + CIRCLE_RADIUS],
                outline=LIGHT_PURPLE,
                width=2
            )

# Try to find a good font
font_paths = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/SFNSDisplay.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
]

font = None
font_size = 95  # Smaller to match original proportions

for font_path in font_paths:
    if os.path.exists(font_path):
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except:
            continue

if font is None:
    font = ImageFont.load_default()

# Draw text - position it to the right of the grid
text = "DEEP VALUE INTELLIGENCE"
text_x = GRID_START_X + GRID_SIZE * CIRCLE_SPACING + 80

# Get text bounding box for vertical centering
bbox = draw.textbbox((0, 0), text, font=font)
text_height = bbox[3] - bbox[1]
text_y = (HEIGHT - text_height) // 2

# Draw the text
draw.text((text_x, text_y), text, font=font, fill=PURPLE)

# Save the image
output_path = os.path.join(os.path.dirname(__file__), "deep_value_intelligence.png")
img.save(output_path, "PNG")
print(f"Logo saved to: {output_path}")
