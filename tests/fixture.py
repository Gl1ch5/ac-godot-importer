"""Генераторы тестовых KN5: машина и трасса.

Формат пишется строго по спецификации Actools, поэтому фикстуры годятся и
для проверки парсера, и для проверки математики (winding / нормали / оси).
"""

from __future__ import annotations

import struct

import numpy as np


# --------------------------------------------------------------------------
# Примитивы
# --------------------------------------------------------------------------

def box(size=(2.0, 1.0, 4.0), center=(0.0, 0.0, 0.0)):
    """Куб с наружными нормалями и CCW-обходом (вид снаружи)."""
    sx, sy, sz = (s * 0.5 for s in size)
    cx, cy, cz = center
    faces = []

    def quad(a, b, c, d, n):
        faces.append(([a, b, c, d], n))

    # +X
    quad((sx, -sy, sz), (sx, -sy, -sz), (sx, sy, -sz), (sx, sy, sz), (1, 0, 0))
    # -X
    quad((-sx, -sy, -sz), (-sx, -sy, sz), (-sx, sy, sz), (-sx, sy, -sz), (-1, 0, 0))
    # +Y
    quad((-sx, sy, sz), (sx, sy, sz), (sx, sy, -sz), (-sx, sy, -sz), (0, 1, 0))
    # -Y
    quad((-sx, -sy, -sz), (sx, -sy, -sz), (sx, -sy, sz), (-sx, -sy, sz), (0, -1, 0))
    # +Z
    quad((-sx, -sy, sz), (sx, -sy, sz), (sx, sy, sz), (-sx, sy, sz), (0, 0, 1))
    # -Z
    quad((sx, -sy, -sz), (-sx, -sy, -sz), (-sx, sy, -sz), (sx, sy, -sz), (0, 0, -1))

    pos, nrm, uv, idx = [], [], [], []
    for fi, (verts, n) in enumerate(faces):
        base = len(pos)
        for k, v in enumerate(verts):
            pos.append((v[0] + cx, v[1] + cy, v[2] + cz))
            nrm.append(n)
            uv.append((0.0 if k in (0, 3) else 1.0, 0.0 if k in (0, 1) else 1.0))
        idx.extend([base, base + 1, base + 2, base, base + 2, base + 3])
    return (
        np.array(pos, dtype=np.float32),
        np.array(nrm, dtype=np.float32),
        np.array(uv, dtype=np.float32),
        np.array(idx, dtype=np.uint16),
    )


def plane(size=(10.0, 10.0), y=0.0, normal=(0, 1, 0), flip=False):
    sx, sz = size[0] * 0.5, size[1] * 0.5
    n = np.array(normal, dtype=np.float32)
    pos = np.array([(-sx, y, sz), (sx, y, sz), (sx, y, -sz), (-sx, y, -sz)], dtype=np.float32)
    nrm = np.tile(n, (4, 1)).astype(np.float32)
    uv = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float32)
    idx = np.array([0, 2, 1, 0, 3, 2] if not flip else [0, 1, 2, 0, 2, 3], dtype=np.uint16)
    return pos, nrm, uv, idx


# --------------------------------------------------------------------------
# DDS (BC1) — чтобы проверить декодер без внешних файлов
# --------------------------------------------------------------------------

def bc1_solid(rgb: tuple[int, int, int], w=8, h=8) -> bytes:
    r, g, b = rgb
    c565 = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
    block = struct.pack("<HHI", c565, 0, 0)
    blocks = (w // 4) * (h // 4)
    payload = block * blocks

    header = bytearray()
    header += b"DDS "
    header += struct.pack("<I", 124)
    header += struct.pack("<I", 0x1 | 0x2 | 0x4 | 0x1000 | 0x80000)  # caps|height|width|pitch|linearsize
    header += struct.pack("<I", h)
    header += struct.pack("<I", w)
    header += struct.pack("<I", len(payload))
    header += struct.pack("<I", 0)
    header += struct.pack("<I", 1)  # mip count
    header += b"\x00" * 44
    header += struct.pack("<I", 32)      # pf size
    header += struct.pack("<I", 0x4)     # DDPF_FOURCC
    header += b"DXT1"
    header += b"\x00" * 20
    header += struct.pack("<I", 0x1000)  # caps
    header += b"\x00" * 16
    return bytes(header) + payload


def bc1_normal(w=8, h=8) -> bytes:
    """Нормаль-мапа: G = 64 (DirectX). После флипа должно стать 191."""
    return bc1_solid((128, 64, 255), w, h)


def bc1_maps(w=8, h=8) -> bytes:
    """txMaps: R=spec, G=gloss, B=metal."""
    return bc1_solid((200, 200, 128), w, h)


# --------------------------------------------------------------------------
# Запись KN5
# --------------------------------------------------------------------------

def _s(text: str) -> bytes:
    raw = text.encode("utf-8") + b"\x00"
    return struct.pack("<i", len(raw)) + raw


class Kn5Writer:
    def __init__(self):
        self.textures: list[tuple[str, bytes]] = []
        self.materials: list[dict] = []

    def texture(self, name: str, data: bytes):
        self.textures.append((name, data))
        return name

    def material(self, name, shader="ksPerPixelMultiMap", blend=0, alpha_tested=False,
                 props=None, maps=None):
        self.materials.append(dict(name=name, shader=shader, blend=blend,
                                   alpha_tested=alpha_tested, props=props or {},
                                   maps=maps or {}))

    def build(self, root: dict, version: int = 5) -> bytes:
        out = bytearray()
        out += b"sc6969"
        out += struct.pack("<i", version)

        out += struct.pack("<i", len(self.textures))
        for name, data in self.textures:
            out += struct.pack("<i", 1)
            out += _s(name)
            out += struct.pack("<I", len(data))
            out += data

        out += struct.pack("<i", len(self.materials))
        for m in self.materials:
            out += _s(m["name"])
            out += _s(m["shader"])
            out += struct.pack("<B", m["blend"])
            out += struct.pack("<B", 1 if m["alpha_tested"] else 0)
            out += struct.pack("<i", 0)
            out += struct.pack("<i", len(m["props"]))
            for pname, val in m["props"].items():
                out += _s(pname)
                out += struct.pack("<f", float(val[0]))
                out += struct.pack("<2f", *val[1])
                out += struct.pack("<3f", *val[2])
                out += struct.pack("<4f", *val[3])
            out += struct.pack("<i", len(m["maps"]))
            slot = 0
            for mname, tex in m["maps"].items():
                out += _s(mname)
                out += struct.pack("<i", slot)
                out += _s(tex)
                slot += 1

        out += self._node(root)
        return bytes(out)

    def _node(self, node: dict) -> bytes:
        cls = node.get("cls", 1)
        out = bytearray()
        out += struct.pack("<i", cls)
        out += _s(node["name"])
        children = node.get("children", [])
        out += struct.pack("<i", len(children))
        out += struct.pack("<B", 1 if node.get("active", True) else 0)

        if cls == 1:
            m = np.asarray(node.get("matrix", np.eye(4)), dtype=np.float64).reshape(4, 4)
            # KN5 хранит A^T в row-major => column-major сериализация A
            out += struct.pack("<16f", *m.flatten(order="F").tolist())
            for c in children:
                out += self._node(c)
            return bytes(out)

        pos, nrm, uv, idx = node["geometry"]
        out += struct.pack("<B", 1)                        # cast shadows
        out += struct.pack("<B", 1 if node.get("visible", True) else 0)
        out += struct.pack("<B", 1 if node.get("transparent", False) else 0)
        out += struct.pack("<I", len(pos))
        for i in range(len(pos)):
            out += struct.pack("<3f", *pos[i])
            out += struct.pack("<3f", *nrm[i])
            out += struct.pack("<2f", *uv[i])
            out += struct.pack("<3f", 0.0, 0.0, 1.0)      # tangent
        out += struct.pack("<I", len(idx))
        out += struct.pack("<%dH" % len(idx), *idx.tolist())
        out += struct.pack("<I", node.get("material", 0))
        out += struct.pack("<I", node.get("layer", 0))
        out += struct.pack("<f", 0.0)
        out += struct.pack("<f", 1e9)
        out += struct.pack("<3f", 0.0, 0.0, 0.0)
        out += struct.pack("<f", 100.0)
        out += struct.pack("<B", 1 if node.get("renderable", True) else 0)
        for c in children:
            out += self._node(c)
        return bytes(out)


# --------------------------------------------------------------------------
# Готовые моды
# --------------------------------------------------------------------------

def make_car_kn5() -> bytes:
    w = Kn5Writer()
    w.texture("body_diffuse.dds", bc1_solid((200, 30, 30)))
    w.texture("body_normal.dds", bc1_normal())
    w.texture("body_maps.dds", bc1_maps())

    props = {
        "ksAmbient": (0.3, (0, 0), (0, 0, 0), (0, 0, 0, 0)),
        "ksDiffuse": (0.9, (0, 0), (0, 0, 0), (0, 0, 0, 0)),
        "ksSpecular": (0.6, (0, 0), (0, 0, 0), (0, 0, 0, 0)),
    }
    maps = {"txDiffuse": "body_diffuse.dds", "txNormal": "body_normal.dds", "txMaps": "body_maps.dds"}
    w.material("body_paint", "ksPerPixelMultiMap", 0, False, props, maps)
    w.material("glass", "ksWindscreen", 1, False, {}, {"txDiffuse": "body_diffuse.dds"})
    w.material("tree_leaf", "ksTree", 0, True, {}, {"txDiffuse": "body_diffuse.dds"})

    wheel_geo = box((0.24, 0.62, 0.62), (0, 0, 0))  # колесо: узкое по X
    body_geo = box((1.9, 0.9, 4.2), (0, 0.55, 0))
    glass_geo = plane((1.4, 1.0), y=1.15)
    tree_geo = plane((3.0, 3.0), y=2.0)

    def wheel(name, x, z):
        return {
            "name": name, "cls": 1, "matrix": _translate(x, 0.33, z),
            "children": [{"name": name + "_MESH", "cls": 2, "geometry": wheel_geo, "material": 0}],
        }

    root = {
        "name": "Root", "cls": 1, "matrix": np.eye(4),
        "children": [
            wheel("WHEEL_LF", -0.82, -1.35),
            wheel("WHEEL_RF", 0.82, -1.35),
            wheel("WHEEL_LR", -0.82, 1.35),
            wheel("WHEEL_RR", 0.82, 1.35),
            {"name": "BODY", "cls": 2, "geometry": body_geo, "material": 0},
            {"name": "GLASS", "cls": 2, "geometry": glass_geo, "material": 1},
            {"name": "TREE", "cls": 2, "geometry": tree_geo, "material": 2},
            {"name": "BLUR_LF", "cls": 2, "geometry": wheel_geo, "material": 0},
            {"name": "COCKPIT_LR", "cls": 2, "geometry": body_geo, "material": 0},
            {"name": "STEER_HR", "cls": 1, "matrix": _translate(-0.35, 0.95, 0.3),
             "children": [{"name": "STEER_HR_MESH", "cls": 2, "geometry": glass_geo, "material": 1}]},
        ],
    }
    return w.build(root)


def make_track_kn5() -> bytes:
    w = Kn5Writer()
    w.texture("road_diffuse.dds", bc1_solid((40, 40, 45)))
    w.material("road_mat", "ksPerPixelMultiMap", 0, False, {},
               {"txDiffuse": "road_diffuse.dds"})
    w.material("grass_mat", "ksPerPixel", 0, False, {}, {"txDiffuse": "road_diffuse.dds"})

    road = plane((12.0, 60.0), y=0.0)
    grass = plane((60.0, 60.0), y=-0.2)
    wall = box((0.3, 1.0, 60.0), (6.5, 0.5, 0))

    root = {
        "name": "Root", "cls": 1, "matrix": np.eye(4),
        "children": [
            {"name": "1ROAD_main", "cls": 2, "geometry": road, "material": 0},
            {"name": "2GRASS_field", "cls": 2, "geometry": grass, "material": 1},
            {"name": "3WALL_left", "cls": 2, "geometry": wall, "material": 0, "renderable": False},
            {"name": "AC_START_0", "cls": 1, "matrix": _translate(0, 0.1, 20.0), "children": []},
            {"name": "AC_PIT_0", "cls": 1, "matrix": _translate(3, 0.1, 25.0), "children": []},
        ],
    }
    return w.build(root)


def _translate(x, y, z) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def write_car_mod(folder: str) -> str:
    import os
    os.makedirs(os.path.join(folder, "data"), exist_ok=True)
    path = os.path.join(folder, "testcar.kn5")
    with open(path, "wb") as f:
        f.write(make_car_kn5())
    with open(os.path.join(folder, "data", "car.ini"), "w", encoding="utf-8") as f:
        f.write("[HEADER]\nVERSION=1.5\n\n[BASIC]\nTOTALMASS=1300\n")
    with open(os.path.join(folder, "data", "lods.ini"), "w", encoding="utf-8") as f:
        f.write("[LOD_0]\nFILE=testcar.kn5\nIN=0\nOUT=1000\n")
    return path


def write_track_mod(folder: str) -> str:
    import os
    os.makedirs(os.path.join(folder, "ai"), exist_ok=True)
    path = os.path.join(folder, "track.kn5")
    with open(path, "wb") as f:
        f.write(make_track_kn5())
    with open(os.path.join(folder, "models.ini"), "w", encoding="utf-8") as f:
        f.write("[MODEL_0]\nFILE=track.kn5\nPOSITION=0,0,0\nROTATION=0,0,0\n")
    return path