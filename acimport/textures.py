"""Декодирование текстур AC (DDS BC1/BC2/BC3/BC4/BC5/BC7 и PNG/JPEG).

AC держит в KN5 «сырые» DDS-файлы (реже PNG/JPEG у CSP-модов). glTF умеет
только PNG/JPEG, поэтому:
  * есть альфа, которую нужно сохранить  -> PNG;
  * непрозрачная                         -> JPEG q=90 (в разы легче);
  * крупные текстуры ресемплируются до max_size (по умолчанию 2048).
"""

from __future__ import annotations

import io
import struct

import numpy as np
from PIL import Image

try:  # необязательная зависимость: без неё останутся только DXT1/DXT5 через PIL
    import texture2ddecoder as _t2d
except Exception:  # pragma: no cover
    _t2d = None

DDS_MAGIC = b"DDS "

# fourCC -> (библиотечная функция, байт на блок)
_FOURCC = {
    b"DXT1": ("decode_bc1", 8),
    b"DXT3": ("decode_bc3", 16),  # BC2: альфа не сжимается, BC3-декодер даёт верный RGB
    b"DXT5": ("decode_bc3", 16),
    b"ATI1": ("decode_bc4", 8),
    b"BC4U": ("decode_bc4", 8),
    b"ATI2": ("decode_bc5", 16),
    b"BC5U": ("decode_bc5", 16),
}

# DXGI_FORMAT -> (функция, байт на блок)
_DXGI = {
    71: ("decode_bc1", 8),
    72: ("decode_bc1", 8),
    74: ("decode_bc3", 16),
    75: ("decode_bc3", 16),
    77: ("decode_bc3", 16),
    78: ("decode_bc3", 16),
    80: ("decode_bc4", 8),
    81: ("decode_bc4", 8),
    83: ("decode_bc5", 16),
    84: ("decode_bc5", 16),
    98: ("decode_bc7", 16),
    99: ("decode_bc7", 16),
}


class TextureError(Exception):
    pass


def _bgra_to_rgba(raw: bytes, w: int, h: int) -> Image.Image:
    """texture2ddecoder отдаёт BGRA."""
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4)
    return Image.fromarray(arr[:, :, [2, 1, 0, 3]], "RGBA")


def decode_dds(data: bytes) -> Image.Image:
    if data[:4] != DDS_MAGIC:
        raise TextureError("не DDS")
    if len(data) < 128:
        raise TextureError("обрезанный DDS-заголовок")

    height, width = struct.unpack_from("<II", data, 12)
    mip_count = struct.unpack_from("<I", data, 28)[0] or 1
    pf_flags = struct.unpack_from("<I", data, 80)[0]
    fourcc = data[84:88]
    offset = 128
    func = block = None

    if pf_flags & 0x4:  # DDPF_FOURCC
        if fourcc == b"DX10":
            if len(data) < 148:
                raise TextureError("обрезанный DX10-заголовок")
            dxgi = struct.unpack_from("<I", data, 128)[0]
            if dxgi not in _DXGI:
                raise TextureError(f"неподдерживаемый DXGI формат {dxgi}")
            func, block = _DXGI[dxgi]
            offset = 148
        elif fourcc in _FOURCC:
            func, block = _FOURCC[fourcc]
        else:
            raise TextureError(f"неподдерживаемый fourCC {fourcc!r}")
    else:
        # несжатый RGB(A) — отдадим PIL, он умеет DDS без сжатия
        img = _pil_dds(data)
        if img is not None:
            return img
        raise TextureError("несжатый DDS без поддержки")

    if _t2d is None:
        img = _pil_dds(data)
        if img is not None:
            return img
        raise TextureError("нужен пакет texture2ddecoder")

    w = max(1, width)
    h = max(1, height)
    need = ((w + 3) // 4) * ((h + 3) // 4) * block
    payload = data[offset : offset + need]
    if len(payload) < need:
        raise TextureError(f"данных меньше ожидаемого ({len(payload)} < {need})")

    fn = getattr(_t2d, func)
    raw = fn(payload, w, h, )
    img = _bgra_to_rgba(raw, w, h)

    if func in ("decode_bc4",):
        # BC4 — одноканальная (обычно «маска»), разложим в серый
        img = img.convert("L").convert("RGBA")
    if mip_count > 1:
        pass  # мипы не нужны: GLB всё равно сгенерирует свои
    return img


def _pil_dds(data: bytes) -> Image.Image | None:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img.convert("RGBA")
    except Exception:
        return None


def load_image(data: bytes) -> Image.Image:
    """Любой блоб из KN5 -> PIL.Image (RGBA)."""
    head = data[:4]
    if head == DDS_MAGIC:
        return decode_dds(data)
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img.convert("RGBA")
    except Exception as exc:  # pragma: no cover
        raise TextureError(f"не удалось декодировать текстуру: {exc}") from exc


def encode_for_glb(
    img: Image.Image,
    keep_alpha: bool,
    max_size: int = 2048,
    quality: int = 90,
) -> tuple[bytes, str]:
    """PIL.Image -> (байты, mime) в формате, понятном glTF."""
    img = img.convert("RGBA") if img.mode != "RGBA" else img
    if max_size and max(img.size) > max_size:
        scale = max_size / max(img.size)
        size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
        img = img.resize(size, Image.LANCZOS)

    buf = io.BytesIO()
    if keep_alpha:
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue(), "image/png"

    rgb = img.convert("RGB")
    img.close()
    rgb.save(buf, format="JPEG", quality=quality)
    return buf.getvalue(), "image/jpeg"


def solid(color: tuple[int, int, int, int] = (255, 255, 255, 255), size: int = 4) -> Image.Image:
    return Image.new("RGBA", (size, size), color)