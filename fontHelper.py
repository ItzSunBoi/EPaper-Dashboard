from __future__ import annotations

import re
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from PIL import Image, ImageFont


# ============================================================
# Parse STM sFONT from .c/.cpp
# ============================================================

# Keeps // headers; removes /* */ blocks (so glyph headers still exist)
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)

# // @3760 '~' (14 pixels wide)
_GLYPH_HEADER_RE = re.compile(
    r"^\s*//\s*@\s*(\d+)\s+'(.{1})'\s*\(.*?\)\s*$",
    re.MULTILINE
)

# sFONT Font16 = { Font16_Table, 11, 16 };
_SFONT_RE = re.compile(
    r"\bsFONT\s+([A-Za-z_]\w*)\s*=\s*\{\s*([A-Za-z_]\w*)\s*,\s*(\d+)\s*,\s*(\d+)\s*,?\s*\}\s*;",
    re.DOTALL
)

# Table initializer: Font16_Table ... = { ... };
_TABLE_RE_TEMPLATE = r"\b{table}\b\s*(?:\[\s*\])?\s*(?:\[\s*\d+\s*\])?\s*=\s*\{{(.*?)\}}\s*;"

# Matches numbers like 0x3F or 123
_NUM_RE = re.compile(r"0x[0-9A-Fa-f]+|\d+")


def _parse_ints(chunk: str) -> List[int]:
    out: List[int] = []
    for tok in _NUM_RE.findall(chunk):
        out.append(int(tok, 16) if tok.lower().startswith("0x") else int(tok))
    return out


def _find_sfont_def(text: str) -> Tuple[str, str, int, int]:
    m = _SFONT_RE.search(text)
    if not m:
        raise ValueError("No sFONT definition found (expected: sFONT FontX = { Table, W, H }; )")
    return m.group(1), m.group(2), int(m.group(3)), int(m.group(4))


def _extract_table_body(text: str, table_name: str) -> str:
    pat = re.compile(_TABLE_RE_TEMPLATE.format(table=re.escape(table_name)), re.DOTALL)
    m = pat.search(text)
    if not m:
        raise ValueError(f"Could not find table initializer for '{table_name}'")
    return m.group(1)


@dataclass(frozen=True)
class STMFontData:
    name: str
    table_name: str
    width: int
    height: int
    bytes_per_row: int
    # ord(char) -> list of bytes length = height * bytes_per_row
    glyph_bytes: Dict[int, List[int]]


def load_stm_sfont_cpp(path: Union[str, Path]) -> STMFontData:
    """
    Loads an STM sFONT (your open-source STM style) from a .c/.cpp file.

    Supports:
      - width <= 8:  1 byte per row
      - width 9..16: 2 bytes per row
      - width 17..24: 3 bytes per row
      etc (bytes_per_row = ceil(width/8))

    Glyphs are discovered by the // @NNN 'C' headers.
    """
    p = Path(path)
    raw = p.read_text(encoding="utf-8", errors="replace")
    raw = re.sub(_BLOCK_COMMENT_RE, "", raw)

    font_name, table_name, width, height = _find_sfont_def(raw)
    table_body = _extract_table_body(raw, table_name)

    bpr = (width + 7) // 8
    glyph_len = height * bpr

    headers = list(_GLYPH_HEADER_RE.finditer(table_body))
    if not headers:
        raise ValueError("No STM glyph headers found (expected lines like: // @3760 '~' (...))")

    glyphs: Dict[int, List[int]] = {}
    for idx, h in enumerate(headers):
        ch = h.group(2)
        code = ord(ch)

        start = h.end()
        end = headers[idx + 1].start() if idx + 1 < len(headers) else len(table_body)
        chunk = table_body[start:end]

        nums = _parse_ints(chunk)
        if len(nums) < glyph_len:
            # Not enough data; skip
            continue

        glyphs[code] = [(n & 0xFF) for n in nums[:glyph_len]]

    if not glyphs:
        raise ValueError("Parsed 0 glyphs. Check headers and table layout.")

    return STMFontData(
        name=font_name,
        table_name=table_name,
        width=width,
        height=height,
        bytes_per_row=bpr,
        glyph_bytes=glyphs,
    )


# ============================================================
# PIL-compatible Font object (no antialiasing)
# ============================================================

class STMSFont(ImageFont.ImageFont):
    """
    Pillow font object backed by STM sFONT data.

    - getmask() returns a binary (0/255) mask -> crisp edges (no AA)
    - draw.text(...) works directly with this font
    """

    def __init__(self, data: STMFontData, hgap: int = 1):
        super().__init__()
        self.data = data
        self.hgap = int(hgap)

    @classmethod
    def from_cpp(cls, path: Union[str, Path], hgap: int = 1) -> "STMSFont":
        return cls(load_stm_sfont_cpp(path), hgap=hgap)

    def getbbox(self, text: str, *args, **kwargs) -> Tuple[int, int, int, int]:
        w, h = self.getsize(text)
        return (0, 0, w, h)

    def getsize(self, text: str, *args, **kwargs) -> Tuple[int, int]:
        if not text:
            return (0, 0)
        w = len(text) * self.data.width + max(0, len(text) - 1) * self.hgap
        h = self.data.height
        return (w, h)

    def getlength(self, text: str) -> float:
        # Pillow may call this for advanced layout; return pixel width.
        return float(self.getsize(text)[0])

    def getmask(self, text: str, mode: str = "L", *args, **kwargs):
        """
        Return an ImagingCore mask. We generate a binary 0/255 'L' mask (no AA).
        """
        mask = self._render_text_mask(text)  # mode 'L'
        return mask.im

    def _render_text_mask(self, text: str) -> Image.Image:
        """
        Creates an 'L' image: 0=transparent, 255=ink. Crisp edges.
        """
        w, h = self.getsize(text)
        w = max(1, w)
        h = max(1, h)
        img = Image.new("L", (w, h), 0)
        px = img.load()

        x_cursor = 0
        d = self.data
        bpr = d.bytes_per_row

        for ch in text:
            code = ord(ch)
            gb = d.glyph_bytes.get(code)

            if gb is None:
                x_cursor += d.width + self.hgap
                continue

            # gb is height*bpr bytes: row0 bytes, row1 bytes, ...
            for yy in range(d.height):
                row_base = yy * bpr
                # Stitch the row bytes into one integer (big-endian)
                row_val = 0
                for bi in range(bpr):
                    row_val = (row_val << 8) | gb[row_base + bi]

                # Bits are MSB-first; we only take 'width' bits from the left.
                # Example width=11, bpr=2 -> row_val is 16 bits, use bits 15..(15-width+1)
                shift = (bpr * 8) - d.width
                row_bits = (row_val >> shift) & ((1 << d.width) - 1)

                for xx in range(d.width):
                    # leftmost pixel is highest remaining bit
                    on = (row_bits >> (d.width - 1 - xx)) & 1
                    if on:
                        px[x_cursor + xx, yy] = 255

            x_cursor += d.width + self.hgap

        return img
