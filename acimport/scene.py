"""Сборка сцен трека и машины в GLB.

Трасса:  визуальные меши батчатся по материалам (100-200 узлов вместо тысяч),
         физические (isRenderable=0 и/или имена *ROAD/*KERB/*GRASS/*SAND/*WALL)
         уходят в отдельный узел Collision с материалами surf_*.
Машина:  иерархия сохраняется (колёса/подвеска/руль), корень поворачивается
         на 180° вокруг Y, машина садится на y=0.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass

import numpy as np

from . import geometry as G
from . import gltf as GL
from . import kn5 as K
from . import materials as M
from . import textures as T


@dataclass
class Options:
    max_texture: int = 2048
    keep_cockpit: bool = False
    keep_blur: bool = False
    merge: bool = True
    quality: int = 90
    no_tangents: bool = False


class MultiTextureCache:
    """Декодирует текстуры из любого из kn5 и кэширует по имени."""

    def __init__(self, kn5s: list[K.Kn5]):
        self.kn5s = kn5s
        self._img: dict[str, object] = {}
        self.errors: list[str] = []

    def get(self, name: str | None):
        if not name:
            return None
        key = name.lower()
        if key in self._img:
            return self._img[key]
        img = None
        for kn5 in self.kn5s:
            tex = kn5.textures.get(key)
            if tex is None:
                continue
            try:
                img = T.load_image(tex.data)
            except Exception as exc:
                self.errors.append(f"{name}: {exc}")
            break
        self._img[key] = img
        return img


# --------------------------------------------------------------------------
# Трассы
# --------------------------------------------------------------------------

SURFACE_RULES = [
    (re.compile(r"ROAD|ASPH|PITLANE|PIT_LANE|^AC_PIT|SKID", re.I), "asphalt"),
    (re.compile(r"KERB|CURB", re.I), "curb"),
    (re.compile(r"GRASS|GREEN", re.I), "grass"),
    (re.compile(r"\bSAND\b|SAND_|_SAND", re.I), "sand"),
    (re.compile(r"GRAVEL|GRAV", re.I), "gravel"),
    (re.compile(r"WALL|BARRIER|ARMCO|GUARD", re.I), "wall"),
    (re.compile(r"CONCRETE|KERBSTONE", re.I), "concrete"),
]

SURFACE_COLORS = {
    "asphalt": (0.06, 0.06, 0.07, 1.0),
    "curb": (0.55, 0.05, 0.05, 1.0),
    "grass": (0.10, 0.22, 0.07, 1.0),
    "sand": (0.55, 0.48, 0.33, 1.0),
    "gravel": (0.35, 0.33, 0.30, 1.0),
    "wall": (0.45, 0.45, 0.47, 1.0),
    "concrete": (0.40, 0.40, 0.40, 1.0),
}


def surface_of(name: str) -> str | None:
    for rx, surf in SURFACE_RULES:
        if rx.search(name):
            return surf
    return None


def collision_material(surf: str) -> M.GltfMaterial:
    color = SURFACE_COLORS.get(surf, (0.2, 0.2, 0.2, 1.0))
    return M.GltfMaterial(
        name=f"surf_{surf}",
        base_color_factor=color,
        metallic_factor=0.0,
        roughness_factor=0.95,
        extras={"surface": surf, "collision": True},
    )


def _node_world(root: K.Node, name: str) -> np.ndarray | None:
    stack = [(root, np.eye(4))]
    while stack:
        node, parent = stack.pop()
        world = parent @ node.transform
        if node.name.upper() == name.upper():
            return world
        for c in node.children:
            stack.append((c, world))
    return None


def build_track(kn5s, builder: GL.GlbBuilder, opts: Options) -> dict:
    """Все kn5 трассы -> один GLB. Возвращает метаданные (спавны, поверхности).

    kn5s: список Kn5 либо пар (Kn5, доп. матрица из models.ini).
    """
    entries: list[tuple[K.Kn5, np.ndarray]] = []
    all_kn5: list[K.Kn5] = []
    for item in kn5s:
        if isinstance(item, tuple):
            entries.append((item[0], np.asarray(item[1], dtype=np.float64)))
            all_kn5.append(item[0])
        else:
            entries.append((item, np.eye(4)))
            all_kn5.append(item)

    root = builder.add_node("Track", matrix=np.eye(4))
    builder.add_root(root)

    cache = MultiTextureCache(all_kn5)
    # есть ли отдельный физический kn5 (physics*.kn5 / fisica*.kn5)
    physics_present = any(
        re.search(r"(physics|fisica|collision|surfaces)", kn5.path, re.I) for kn5 in all_kn5
    )
    DRIVEABLE = {"asphalt", "curb", "grass", "sand", "gravel", "wall", "concrete"}
    visual: dict[str, list[tuple]] = {}
    material_of: dict[str, K.Material] = {}
    collision: dict[str, list[tuple]] = {}
    info = {"start": None, "pit": None, "surfaces": {}, "skipped": [], "mesh_nodes": 0,
            "source_meshes": 0}

    for kn5, extra in entries:
        if kn5.root is None:
            continue
        for node, _path, world in K.iter_meshes(kn5.root):
            if node.vertices is None or len(node.indices) < 3 or not node.active:
                continue
            info["source_meshes"] += 1
            mat = kn5.material_by_id(node.material)
            pos, nrm, uv, idx = G.bake(node, extra @ world)
            surf = surface_of(node.name)

            # физическая сетка: явно невидимая или меш из physics-kn5 с именем поверхности
            if (not node.renderable) or (physics_present and surf):
                collision.setdefault(surf or "asphalt", []).append((pos, nrm, uv, idx))
                if not node.renderable:
                    continue
            elif not node.visible or mat is None:
                info["skipped"].append(node.name)
                continue
            else:
                # физического kn5 в моде нет — коллизию строим по дорожным мешам
                if surf in DRIVEABLE:
                    collision.setdefault(surf, []).append((pos, nrm, uv, idx))

            visual.setdefault(mat.name, []).append((pos, nrm, uv, idx))
            material_of[mat.name] = mat

        for special, key in (("AC_START_0", "start"), ("AC_PIT_0", "pit")):
            if info[key] is None:
                w = _node_world(kn5.root, special)
                if w is not None:
                    info[key] = [round(float(v), 4) for v in w[:3, 3]]

    # --- визуал: один узел на материал ---
    for name, parts in sorted(visual.items()):
        merged = G.merge_arrays(parts)
        if merged is None:
            continue
        pos, nrm, uv, idx = merged
        gm = M.convert(material_of[name], cache)
        mi = builder.add_material(gm, opts.max_texture)
        prim = GL.Primitive(
            positions=pos.astype(np.float32),
            normals=nrm.astype(np.float32),
            uvs=uv.astype(np.float32) if uv is not None else None,
            tangents=(G.compute_tangents(pos, nrm, uv, idx)
                      if (uv is not None and not opts.no_tangents) else None),
            indices=idx.astype(np.uint32),
            material=mi,
        )
        mesh = builder.add_mesh(f"vis_{name}", [prim])
        ni = builder.add_node(f"vis_{name}", mesh=mesh, matrix=np.eye(4))
        builder.nodes[root].setdefault("children", []).append(ni)
        info["mesh_nodes"] += 1

    # --- коллизия ---
    if collision:
        col_idx = builder.add_node("Collision", matrix=np.eye(4))
        builder.nodes[root].setdefault("children", []).append(col_idx)
        for surf, parts in sorted(collision.items()):
            merged = G.merge_arrays(parts)
            if merged is None:
                continue
            pos, nrm, uv, idx = merged
            mi = builder.add_material(collision_material(surf), opts.max_texture)
            prim = GL.Primitive(
                positions=pos.astype(np.float32),
                normals=nrm.astype(np.float32),
                uvs=None,
                tangents=None,
                indices=idx.astype(np.uint32),
                material=mi,
            )
            mesh = builder.add_mesh(f"surf_{surf}", [prim])
            ni = builder.add_node(f"surf_{surf}", mesh=mesh, matrix=np.eye(4))
            builder.nodes[col_idx].setdefault("children", []).append(ni)
            info["surfaces"][surf] = int(len(idx) // 3)

    info["texture_errors"] = cache.errors
    return info


# --------------------------------------------------------------------------
# Машины
# --------------------------------------------------------------------------

CAR_EXCLUDE = [
    "BLUR_*", "RIM_BLUR_*", "*_BLUR", "COCKPIT_LR", "COLLIDER*", "AC_COLLIDER*",
    "SHADOW*", "BBOX*", "LOD_*", "COLLISION*", "*_LOD1", "*_LOD2", "*_LOD3",
]

# поворот на 180° вокруг Y: определитель +1, обход треугольников не меняется
ROT180 = np.array(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _excluded(name: str, opts: Options) -> bool:
    up = name.upper()
    if opts.keep_cockpit and up.startswith("COCKPIT_LR"):
        return False
    if opts.keep_blur and "BLUR" in up:
        return False
    return any(fnmatch.fnmatch(up, pat) for pat in CAR_EXCLUDE)


def build_car(kn5: K.Kn5, builder: GL.GlbBuilder, opts: Options) -> dict:
    """KN5 машины -> GLB с поворотом 180° по Y и посадкой на y=0."""
    if kn5.root is None:
        raise ValueError("пустой KN5")

    car = builder.add_node("Car", matrix=np.eye(4))
    builder.add_root(car)

    mat_cache: dict[str, int] = {}
    cache = MultiTextureCache([kn5])
    stats = {"nodes": 0, "meshes": 0, "skipped": [], "triangles": 0}

    def convert(node: K.Node, parent_idx: int) -> None:
        if _excluded(node.name, opts):
            stats["skipped"].append(node.name)
            return

        if node.is_mesh:
            if node.vertices is None or len(node.indices) < 3 or not node.renderable or not node.visible:
                stats["skipped"].append(node.name)
                return
            mat = kn5.material_by_id(node.material)
            if mat is None:
                stats["skipped"].append(node.name)
                return
            pos, nrm, uv, idx = G.bake(node, node.transform)
            shader = (mat.shader or "").lower()
            if "tree" in shader or "grass" in shader:
                nrm = G.fix_tree_normals(nrm)
            gm = M.convert(mat, cache)
            if gm.name not in mat_cache:
                mat_cache[gm.name] = builder.add_material(gm, opts.max_texture)
            prim = GL.Primitive(
                positions=pos.astype(np.float32),
                normals=nrm.astype(np.float32),
                uvs=uv.astype(np.float32) if uv is not None else None,
                tangents=G.compute_tangents(pos, nrm, uv, idx) if uv is not None else None,
                indices=idx.astype(np.uint32),
                material=mat_cache[gm.name],
            )
            mesh = builder.add_mesh(node.name, [prim])
            idx_node = builder.add_node(node.name, mesh=mesh)
            builder.nodes[parent_idx].setdefault("children", []).append(idx_node)
            stats["meshes"] += 1
            stats["nodes"] += 1
            stats["triangles"] += len(idx) // 3
            for c in node.children:
                convert(c, idx_node)
            return

        idx_node = builder.add_node(node.name, matrix=node.transform)
        builder.nodes[parent_idx].setdefault("children", []).append(idx_node)
        stats["nodes"] += 1
        for c in node.children:
            convert(c, idx_node)

    convert(kn5.root, car)

    # --- колесо/руль в мировых координатах (уже после поворота) ---
    wheels = {}
    for name in ("WHEEL_LF", "WHEEL_RF", "WHEEL_LR", "WHEEL_RR"):
        w = _node_world(kn5.root, name)
        if w is None:
            continue
        p = (ROT180 @ w)[:3, 3]
        wheels[name] = [round(float(v), 4) for v in p]

    steer = None
    for cand in ("STEER_HR", "STEER_LR", "STEER"):
        if kn5.root.find(cand) is not None:
            steer = cand
            break

    # --- посадка на землю ---
    min_y = None
    for node, _path, world in K.iter_meshes(kn5.root):
        if not node.renderable or node.vertices is None or not node.visible:
            continue
        pos, _, _, _ = G.bake(node, world)
        if len(pos) == 0:
            continue
        low = float(pos[:, 1].min())
        min_y = low if min_y is None else min(min_y, low)

    m = ROT180.copy()
    m[1, 3] = -(min_y or 0.0)
    builder.nodes[car]["matrix"] = [round(float(v), 7) for v in m.flatten(order="F")]

    return {
        "wheels": wheels,
        "steer": steer,
        "min_y_before_offset": round(float(min_y), 5) if min_y is not None else None,
        "stats": stats,
        "texture_errors": cache.errors,
    }


def wheel_layout_report(wheels: dict[str, list[float]]) -> dict:
    """Проверка, что WHEEL_* стоят в правильных квадрантах Godot-системы."""
    expect = {
        "WHEEL_LF": lambda x, z: x < 0 and z < 0,
        "WHEEL_RF": lambda x, z: x > 0 and z < 0,
        "WHEEL_LR": lambda x, z: x < 0 and z > 0,
        "WHEEL_RR": lambda x, z: x > 0 and z > 0,
    }
    out = {}
    for name, pred in expect.items():
        p = wheels.get(name)
        if p is None:
            out[name] = "отсутствует"
        else:
            out[name] = "ok" if pred(p[0], p[2]) else f"НЕВЕРНО ({p[0]:.2f}, {p[2]:.2f})"
    return out