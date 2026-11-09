"""Минималистичный, но корректный сборщик бинарного GLB (glTF 2.0).

Без внешних зависимостей: json + struct + numpy. На выходе один файл .glb
(JSON-чанк + BIN-чанк), который Godot 4 импортирует как обычную сцену.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from . import textures as T

FLOAT = 5126
UINT = 5125
USHORT = 5123
ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963

# glTF требует выравнивания bufferView по 4 байта
_ALIGN = 4


class GltfError(Exception):
    pass


@dataclass
class Primitive:
    positions: np.ndarray          # (N,3) float32
    normals: np.ndarray | None     # (N,3) float32
    uvs: np.ndarray | None         # (N,2) float32
    tangents: np.ndarray | None    # (N,4) float32
    indices: np.ndarray            # (M,) uint32
    material: int = 0
    min_xyz: tuple | None = None
    max_xyz: tuple | None = None


class GlbBuilder:
    def __init__(self, generator: str = "ac_importer"):
        self.generator = generator
        self._bin = bytearray()
        self.accessors: list[dict] = []
        self.buffer_views: list[dict] = []
        self.meshes: list[dict] = []
        self.nodes: list[dict] = []
        self.materials: list[dict] = []
        self.images: list[dict] = []
        self.textures: list[dict] = []
        self.samplers: list[dict] = []
        self.scene_nodes: list[int] = []
        self.extensions_used: set[str] = set()
        self._image_cache: dict[bytes, int] = {}
        self._tex_cache: dict[tuple, int] = {}
        self._mat_cache: dict[str, int] = {}

    # ------------------------------------------------------------------ bin
    def _align(self) -> None:
        while len(self._bin) % _ALIGN:
            self._bin.append(0)

    def _view(self, data: bytes, target: int | None = None) -> int:
        self._align()
        offset = len(self._bin)
        self._bin.extend(data)
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        self.buffer_views.append(view)
        return len(self.buffer_views) - 1

    def _accessor(self, arr: np.ndarray, ctype: int, atype: str, target: int,
                  minmax: bool = False) -> int:
        arr = np.ascontiguousarray(arr)
        view = self._view(arr.tobytes(), target)
        acc = {
            "bufferView": view,
            "componentType": ctype,
            "count": int(arr.shape[0]),
            "type": atype,
        }
        if minmax:
            acc["min"] = [float(v) for v in np.min(arr, axis=0)]
            acc["max"] = [float(v) for v in np.max(arr, axis=0)]
        self.accessors.append(acc)
        return len(self.accessors) - 1

    # ------------------------------------------------------------- images
    def add_image(self, img: Image.Image, keep_alpha: bool, max_size: int = 2048,
                  quality: int = 90) -> int:
        data, mime = T.encode_for_glb(img, keep_alpha, max_size, quality)
        key = (data, mime)
        cached = self._image_cache.get(data)
        if cached is not None:
            return self._texture_for(cached, mime)
        view = self._view(data)
        self.images.append({"bufferView": view, "mimeType": mime})
        idx = len(self.images) - 1
        self._image_cache[data] = idx
        return self._texture_for(idx, mime, data)

    def _texture_for(self, image_idx: int, mime: str, data: bytes | None = None) -> int:
        key = (image_idx,)
        if key in self._tex_cache:
            return self._tex_cache[key]
        if not self.samplers:
            self.samplers.append({"magFilter": 9729, "minFilter": 9987,
                                  "wrapS": 10497, "wrapT": 10497})
        self.textures.append({"sampler": 0, "source": image_idx})
        idx = len(self.textures) - 1
        self._tex_cache[key] = idx
        return idx

    # ---------------------------------------------------------- materials
    def add_material(self, gm, max_size: int = 2048) -> int:
        if gm.name in self._mat_cache:
            return self._mat_cache[gm.name]
        keep_alpha = gm.alpha_mode != "OPAQUE"
        pbr: dict = {
            "baseColorFactor": [round(float(v), 5) for v in gm.base_color_factor],
            "metallicFactor": round(float(gm.metallic_factor), 4),
            "roughnessFactor": round(float(gm.roughness_factor), 4),
        }
        if gm.base_color_texture is not None:
            pbr["baseColorTexture"] = {"index": self.add_image(gm.base_color_texture, keep_alpha, max_size)}
        if gm.metallic_texture is not None:
            # PNG: JPEG портит каналы шероховатости/металла
            pbr["metallicRoughnessTexture"] = {"index": self.add_image(gm.metallic_texture, True, max_size)}

        mat: dict = {
            "name": gm.name,
            "pbrMetallicRoughness": pbr,
            "alphaMode": gm.alpha_mode,
            "doubleSided": bool(gm.double_sided),
        }
        if gm.alpha_mode == "MASK":
            mat["alphaCutoff"] = round(float(gm.alpha_cutoff), 3)
        if gm.normal_texture is not None:
            mat["normalTexture"] = {
                "index": self.add_image(gm.normal_texture, False, max_size),
                "scale": round(float(gm.normal_scale), 4),
            }
        if gm.emissive_texture is not None:
            mat["emissiveTexture"] = {"index": self.add_image(gm.emissive_texture, False, max_size)}
        if any(v > 0 for v in gm.emissive_factor):
            mat["emissiveFactor"] = [round(float(v), 4) for v in gm.emissive_factor]
        if gm.is_paint:
            mat["extensions"] = {
                "KHR_materials_clearcoat": {"clearcoatFactor": 1.0, "clearcoatRoughnessFactor": 0.08}
            }
            self.extensions_used.add("KHR_materials_clearcoat")
        if gm.extras:
            mat["extras"] = gm.extras
        self.materials.append(mat)
        idx = len(self.materials) - 1
        self._mat_cache[gm.name] = idx
        return idx

    # -------------------------------------------------------------- meshes
    def add_mesh(self, name: str, prims: list[Primitive]) -> int:
        gl_prims = []
        for p in prims:
            attrs = {}
            pos = p.positions.astype(np.float32)
            attrs["POSITION"] = self._accessor(pos, FLOAT, "VEC3", ARRAY_BUFFER, minmax=True)
            if p.normals is not None and len(p.normals):
                attrs["NORMAL"] = self._accessor(p.normals.astype(np.float32), FLOAT, "VEC3", ARRAY_BUFFER)
            if p.uvs is not None and len(p.uvs):
                attrs["TEXCOORD_0"] = self._accessor(p.uvs.astype(np.float32), FLOAT, "VEC2", ARRAY_BUFFER)
            if p.tangents is not None and len(p.tangents):
                attrs["TANGENT"] = self._accessor(p.tangents.astype(np.float32), FLOAT, "VEC4", ARRAY_BUFFER)

            idx = p.indices.astype(np.uint32)
            if len(idx) and int(idx.max()) < 65535:
                ctype, cast = USHORT, idx.astype(np.uint16)
            else:
                ctype, cast = UINT, idx
            attrs_idx = self._accessor(cast, ctype, "SCALAR", ELEMENT_ARRAY_BUFFER)

            prim = {"attributes": attrs, "indices": attrs_idx, "mode": 4}
            if p.material >= 0:
                prim["material"] = p.material
            gl_prims.append(prim)

        self.meshes.append({"name": name, "primitives": gl_prims})
        return len(self.meshes) - 1

    # --------------------------------------------------------------- nodes
    def add_node(self, name: str, mesh: int | None = None, matrix: np.ndarray | None = None,
                 translation=None, rotation=None, scale=None,
                 children: list[int] | None = None) -> int:
        node: dict = {"name": name}
        if matrix is not None:
            # матрица принимается в обычной аффинной форме (column-vector),
            # glTF ждёт column-major массив => flatten(order="F")
            arr = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
            node["matrix"] = [round(float(v), 7) for v in arr.flatten(order="F")]
        else:
            if translation is not None:
                node["translation"] = [round(float(v), 7) for v in translation]
            if rotation is not None:
                node["rotation"] = [round(float(v), 7) for v in rotation]
            if scale is not None:
                node["scale"] = [round(float(v), 7) for v in scale]
        if mesh is not None:
            node["mesh"] = mesh
        if children:
            node["children"] = list(children)
        self.nodes.append(node)
        return len(self.nodes) - 1

    def add_root(self, idx: int) -> None:
        self.scene_nodes.append(idx)

    # ---------------------------------------------------------------- save
    def to_glb(self) -> bytes:
        doc = {
            "asset": {"version": "2.0", "generator": self.generator},
            "scene": 0,
            "scenes": [{"name": "Scene", "nodes": self.scene_nodes}],
            "nodes": self.nodes,
            "meshes": self.meshes,
            "accessors": self.accessors,
            "bufferViews": self.buffer_views,
            "buffers": [{"byteLength": len(self._bin)}],
        }
        if self.materials:
            doc["materials"] = self.materials
        if self.images:
            doc["images"] = self.images
            doc["textures"] = self.textures
            doc["samplers"] = self.samplers
        if self.extensions_used:
            doc["extensionsUsed"] = sorted(self.extensions_used)

        js = json.dumps(doc, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        js += b" " * ((4 - len(js) % 4) % 4)
        bin_chunk = bytes(self._bin)
        bin_chunk += b"\x00" * ((4 - len(bin_chunk) % 4) % 4)

        total = 12 + 8 + len(js) + (8 + len(bin_chunk) if bin_chunk else 0)
        out = bytearray()
        out += struct.pack("<4sII", b"glTF", 2, total)
        out += struct.pack("<I4s", len(js), b"JSON")
        out += js
        if bin_chunk:
            out += struct.pack("<I4s", len(bin_chunk), b"BIN\x00")
            out += bin_chunk
        return bytes(out)

    def save(self, path: str) -> int:
        data = self.to_glb()
        with open(path, "wb") as f:
            f.write(data)
        return len(data)

    # ------------------------------------------------------------- helpers
    def summary(self) -> str:
        verts = 0
        tris = 0
        for m in self.meshes:
            for p in m["primitives"]:
                acc = self.accessors[p["attributes"]["POSITION"]]
                verts += acc["count"]
                tris += self.accessors[p["indices"]]["count"] // 3
        return (f"узлов {len(self.nodes)}, мешей {len(self.meshes)}, материалов {len(self.materials)}, "
                f"текстур {len(self.textures)}, вершин {verts}, треугольников {tris}")