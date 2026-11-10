"""Перевод материалов Assetto Corsa в PBR glTF 2.0.

Правила (из ТЗ и практики AC/CSP):

BLEND MODE
  * 0 (Opaque) и не alpha_tested, не стекло -> alphaMode=OPAQUE, альфа-канал
    txDiffuse ВЫБРАСЫВАЕТСЯ. В AC там лежит specular-маска: если её оставить,
    кузов/асфальт в Godot станут полупрозрачными.
  * alpha_tested или blend=2 (alpha-to-coverage) или шейдер ksTree -> MASK,
    alphaCutoff=0.4.
  * 1 (AlphaBlend) -> BLEND, doubleSided=false (стёкла, фары, тени, groove).

ТЕКСТУРЫ
  * txDiffuse   -> baseColorTexture (RGB или RGBA).
  * txNormal    -> normalTexture, зелёный канал инвертируется (AC = DirectX Y-).
  * txMaps      -> metallicRoughnessTexture: G = 255 - G(gloss), B = B(metal).
  * txEmissive  -> emissiveTexture.
  * txDetail / txNormalDetail — в PBR не переносятся (детализация AC-шейдера),
    их имена сохраняются в extras на случай ручной доработки.

ДЕРЕВЬЯ
  * ksTree: нормали вершин пересчитываются в конвертере (см. geometry.py),
    материал помечается doubleSided.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from . import kn5 as K
from . import textures as T

# шейдеры, у которых альфа в txDiffuse — реальная прозрачность
ALPHA_SHADERS = {"kswindscreen", "ksperpixelalpha", "ksbrakedisc", "ksperpixelalpha"}

PAINT_NAME_HINTS = ("carpaint", "body", "ext_body", "skin", "paint")

ALPHA_MODE_OPAQUE = "OPAQUE"
ALPHA_MODE_MASK = "MASK"
ALPHA_MODE_BLEND = "BLEND"


@dataclass
class GltfMaterial:
    name: str
    alpha_mode: str = ALPHA_MODE_OPAQUE
    alpha_cutoff: float = 0.4
    double_sided: bool = False
    base_color_texture: Image.Image | None = None
    base_color_factor: tuple = (1.0, 1.0, 1.0, 1.0)
    metallic_texture: Image.Image | None = None
    metallic_factor: float = 1.0
    roughness_factor: float = 1.0
    normal_texture: Image.Image | None = None
    normal_scale: float = 1.0
    emissive_texture: Image.Image | None = None
    emissive_factor: tuple = (0.0, 0.0, 0.0)
    is_paint: bool = False
    extras: dict = field(default_factory=dict)

    def key(self) -> str:
        """Хэш визуального состояния: одинаковые материалы переиспользуются."""
        h = hashlib.sha1()
        h.update(self.name.encode())
        for img in (self.base_color_texture, self.metallic_texture, self.normal_texture, self.emissive_texture):
            if img is not None:
                h.update(img.tobytes())
        h.update(repr((self.alpha_mode, self.alpha_cutoff, self.double_sided,
                       self.base_color_factor, self.metallic_factor, self.roughness_factor,
                       self.normal_scale, self.emissive_factor)).encode())
        return h.hexdigest()


class TextureCache:
    """Декодирование каждого блоба ровно один раз."""

    def __init__(self, kn5: K.Kn5):
        self.kn5 = kn5
        self._img: dict[str, Image.Image | None] = {}
        self.errors: list[str] = []

    def get(self, name: str | None) -> Image.Image | None:
        if not name:
            return None
        key = name.lower()
        if key in self._img:
            return self._img[key]
        tex = self.kn5.textures.get(key)
        if tex is None:
            self._img[key] = None
            return None
        try:
            img = T.load_image(tex.data)
        except Exception as exc:
            self.errors.append(f"{name}: {exc}")
            img = None
        self._img[key] = img
        return img


def _flip_normal(img: Image.Image) -> Image.Image:
    """DirectX (Y-) -> OpenGL (Y+), затем чистый RGB."""
    rgb = img.convert("RGB")
    arr = np.array(rgb, dtype=np.uint8)
    arr[:, :, 1] = 255 - arr[:, :, 1]
    return Image.fromarray(arr, "RGB")


def _metallic_roughness(img: Image.Image, is_track: bool = False) -> Image.Image:
    """txMaps (R=spec, G=gloss, B=metal) -> glTF metallicRoughness (G=rough, B=metal)."""
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    out = np.empty_like(arr)
    out[:, :, 0] = 255                      # R не используется glTF; держим белым
    rough = 255 - arr[:, :, 1]              # G = 255 - gloss = roughness
    if is_track:
        # Для трасс зажимаем минимальную шероховатость (мировые дороги не зеркалят скайбокс)
        rough = np.clip(rough, 180, 255)
    out[:, :, 1] = rough
    out[:, :, 2] = arr[:, :, 2]             # B = metal
    return Image.fromarray(out, "RGB")


def _has_alpha(img: Image.Image, threshold: int = 3) -> bool:
    if img.mode not in ("RGBA", "LA"):
        return False
    a = np.array(img.getchannel("A"))
    return bool((a < 255 - threshold).any())


def _bake_multilayer_color(diffuse: Image.Image | None, mask: Image.Image | None,
                           details: list[Image.Image | None], size: int = 1024) -> Image.Image | None:
    """Для glTF: комбинирует macro txDiffuse и средние цвета txDetail по маске txMask."""
    if diffuse is None:
        # Если txDiffuse нет, берем первый непустой detail
        for d in details:
            if d is not None:
                return d.convert("RGB")
        return None

    diff_rgb = diffuse.convert("RGB")
    valid_details = [d for d in details if d is not None]
    if not valid_details:
        return diff_rgb

    # Считаем средний RGB цвет для каждого слоя
    means = []
    for d in details:
        if d is not None:
            arr = np.asarray(d.convert("RGB"), dtype=np.float32) / 255.0
            means.append(arr.mean(axis=(0, 1)))
        else:
            means.append(np.array([0.5, 0.5, 0.5], dtype=np.float32))

    if mask is None:
        # Усредняем по всем слоям
        avg_mean = np.mean(means, axis=0)
        diff_arr = np.asarray(diff_rgb, dtype=np.float32) / 255.0
        baked = np.clip(diff_arr * avg_mean * 1.8 * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(baked, "RGB")

    # С маской
    m_img = mask.convert("RGBA").resize(diff_rgb.size, Image.Resampling.BILINEAR)
    m_arr = np.asarray(m_img, dtype=np.float32) / 255.0
    weights = m_arr[:, :, :4] # R, G, B, A
    w_sum = weights.sum(axis=2, keepdims=True)
    zero_mask = (w_sum < 0.01).squeeze(axis=2)
    w_sum = np.maximum(w_sum, 0.001)
    norm_w = weights / w_sum

    # Комбинируем средние цвета деталей по пикселям маски
    detail_col = np.zeros_like(norm_w[:, :, :3])
    for ch in range(4):
        detail_col += norm_w[:, :, ch:ch+1] * means[ch]
    # Где маска нулевая — берем дефолтный серый оттенок деталей
    detail_col[zero_mask] = np.mean(means, axis=0)

    diff_arr = np.asarray(diff_rgb, dtype=np.float32) / 255.0
    baked = np.clip(diff_arr * detail_col * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(baked, "RGB")


def is_paint(name: str) -> bool:
    low = name.lower()
    return any(h in low for h in PAINT_NAME_HINTS)


CUTOUT_HINTS = (
    "fence", "wire", "tree", "leaf", "leaves", "branch", "foliage", "bark",
    "bush", "grass", "net", "chain", "grate", "grille", "mesh", "rail",
    "canvas", "canopy", "stencil", "billboard", "flag", "banner", "crowd",
    "hedge", "shrub", "plant", "fern", "ivy", "creeper", "straw", "guardrail",
    "barrier", "brabham", "stand", "seat", "box", "building", "glass", "window"
)


def _is_cutout_candidate(mat_name: str, tex_name: str | None) -> bool:
    s = (mat_name + " " + (tex_name or "")).lower()
    return any(h in s for h in CUTOUT_HINTS)


def _detect_alpha_cutout(img: Image.Image | None) -> bool:
    """Определяет, есть ли в альфа-канале текстуры реальные вырезы (alpha < 128)."""
    if img is None or img.mode not in ("RGBA", "LA"):
        return False
    try:
        a = np.array(img.getchannel("A"))
        # если есть хотя бы 0.5% пикселей с прозрачностью < 128 и есть непрозрачные пиксели
        zeros = float(np.mean(a < 128))
        return 0.005 < zeros < 0.99
    except Exception:
        return False


def convert(material: K.Material, cache: TextureCache, is_track: bool = False) -> GltfMaterial:
    shader = (material.shader or "").lower().replace("\\", "/").split("/")[-1]
    shader = shader.replace(".fx", "")
    alpha_tested = bool(material.alpha_tested)
    blend = material.blend_mode
    is_tree = shader in ("kstree", "kstree_nodamage") or "tree" in material.name.lower()
    is_glass = shader in ALPHA_SHADERS

    gm = GltfMaterial(name=material.name, is_paint=is_paint(material.name))

    diffuse = cache.get(material.texture("txDiffuse"))

    # Если это ksMultilayer: запекаем цвета деталей с макро-текстурой
    if "multilayer" in shader:
        mask = cache.get(material.texture("txMask"))
        det_r = cache.get(material.texture("txDetailR"))
        det_g = cache.get(material.texture("txDetailG"))
        det_b = cache.get(material.texture("txDetailB"))
        det_a = cache.get(material.texture("txDetailA"))
        diffuse = _bake_multilayer_color(diffuse, mask, [det_r, det_g, det_b, det_a])

    # --- альфа ---
    is_decal = any(kw in material.name.lower() for kw in ("groove", "skid", "decal", "mark", "line", "rubber")) and "trees" not in material.name.lower()
    is_tree = (
        shader in ("kstree", "kstree_nodamage", "ksgrass")
        or any(kw in material.name.lower() for kw in ("tree", "bush", "foliage", "pine", "hedge", "leaf", "leaves", "branch", "wood", "treeline", "treesline"))
    )
    if is_track:
        # Для трасс:
        if is_tree:
            gm.alpha_mode = ALPHA_MODE_MASK
            gm.alpha_cutoff = 0.5
            gm.double_sided = True
        elif alpha_tested or blend == 2 or shader.endswith("at") or shader.endswith("at_ns"):
            gm.alpha_mode = ALPHA_MODE_MASK
            ref = _prop_scalar(material, "ksAlphaRef", 0.35)
            gm.alpha_cutoff = float(np.clip(ref * 0.65, 0.12, 0.5))
            gm.double_sided = True
        # 2. Декали и накат — строго BLEND (не MASK!)
        elif shader in ("ksperpixelalpha", "ksskidmarks", "kswindscreen", "ksclouds") or is_decal or blend == 1:
            gm.alpha_mode = ALPHA_MODE_BLEND
            gm.double_sided = False
        else:
            gm.alpha_mode = ALPHA_MODE_OPAQUE
    else:
        # Для машин: классический пайплайн — сохраняем альфа-режимы и спекуляр-маски авто
        tex_name = material.texture("txDiffuse")
        is_cutout = _is_cutout_candidate(material.name, tex_name) and _detect_alpha_cutout(diffuse)
        if is_tree or alpha_tested or blend == 2 or is_cutout:
            gm.alpha_mode = ALPHA_MODE_MASK
            gm.alpha_cutoff = 0.4
            gm.double_sided = True
        elif blend == 1 or is_glass:
            gm.alpha_mode = ALPHA_MODE_BLEND
            gm.double_sided = False
        else:
            gm.alpha_mode = ALPHA_MODE_OPAQUE

    if is_tree:
        gm.double_sided = True

    # --- базовый цвет ---
    if is_track and any(kw in material.name.lower() for kw in ("asph", "road", "track", "tarmac", "sand", "gravel", "runoff", "apsh", "top_b", "line")):
        # Принудительно гасим отражение неба и пересвет на дорогах и зонах вылета
        gm.base_color_factor = (0.55, 0.55, 0.55, 1.0)
        gm.roughness_factor = 0.98
        gm.metallic_factor = 0.0
    elif is_track and is_decal:
        # Для дорожных декалей (groove / skid) умножаем на темный коэффициент
        gm.base_color_factor = (0.2, 0.2, 0.2, 1.0)
        gm.roughness_factor = 0.98
        gm.metallic_factor = 0.0
    else:
        gm.base_color_factor = (1.0, 1.0, 1.0, 1.0)
    if diffuse is not None:
        if gm.alpha_mode == ALPHA_MODE_OPAQUE and is_track:
            # Для трасс принудительно сбрасываем в RGB
            gm.base_color_texture = diffuse.convert("RGB")
        elif gm.alpha_mode == ALPHA_MODE_OPAQUE and not is_glass:
            gm.base_color_texture = diffuse.convert("RGB")
        else:
            gm.base_color_texture = diffuse.convert("RGBA")

    # --- металл/шероховатость ---
    maps = cache.get(material.texture("txMaps"))
    is_ground = is_track and any(kw in material.name.lower() for kw in ("asph", "road", "track", "tarmac", "sand", "gravel", "runoff", "apsh", "top_b", "grass"))
    if maps is not None and not is_ground:
        gm.metallic_texture = _metallic_roughness(maps, is_track=is_track)
        gm.metallic_factor = max(0.0, min(1.0, _prop_scalar(material, "ksSpecular", 1.0)))
    else:
        spec = _prop_scalar(material, "ksSpecular", 0.05)
        gm.metallic_factor = 0.0
        if is_track:
            # Для трасс зажимаем матовость и убираем спекуляр полностью
            gm.roughness_factor = 0.98
            gm.metallic_factor = 0.0
        else:
            gm.roughness_factor = float(np.clip(1.0 - spec, 0.08, 1.0))

    # --- нормали ---
    # Для дорог и зон вылета на трассах отключаем Normal map, чтобы не было сиренево-голубого Z-fighting муара
    if is_track and any(kw in material.name.lower() for kw in ("asph", "road", "track", "tarmac", "sand", "gravel", "runoff", "apsh", "top_b", "line", "groove")):
        pass
    else:
        nrm = cache.get(material.texture("txNormal"))
        if nrm is not None:
            gm.normal_texture = _flip_normal(nrm)
        elif not is_track:
            nrm_det = cache.get(material.texture("txNormalDetail"))
            if nrm_det is not None:
                gm.normal_texture = _flip_normal(nrm_det)

    # --- эмиссия ---
    emis = cache.get(material.texture("txEmissive"))
    emis_prop = material.prop("ksEmissive")
    if emis is not None:
        gm.emissive_texture = emis.convert("RGB")
        gm.emissive_factor = (1.0, 1.0, 1.0)
    elif emis_prop is not None and max(emis_prop.c) > 0:
        gm.emissive_factor = tuple(float(min(1.0, v)) for v in emis_prop.c)

    gm.extras = {
        "ac_shader": material.shader,
        "ac_blend_mode": blend,
        "ac_alpha_tested": alpha_tested,
        "ac_depth_mode": material.depth_mode,
        "ac_textures": {m.name: m.texture for m in material.mappings},
        "ac_properties": {
            p.name: {"a": p.a, "b": list(p.b), "c": list(p.c), "d": list(p.d)}
            for p in material.properties.values()
        },
        "clearcoat": bool(gm.is_paint),
    }
    return gm


def _prop_scalar(material: K.Material, name: str, default: float) -> float:
    p = material.prop(name)
    return float(p.a) if p is not None else default


def build_all(kn5: K.Kn5) -> tuple[list[GltfMaterial], list[str]]:
    cache = TextureCache(kn5)
    out = [convert(m, cache) for m in kn5.materials]
    return out, cache.errors