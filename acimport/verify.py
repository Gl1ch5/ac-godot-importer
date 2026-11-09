"""Проверка собранного GLB без Godot: структура, байты, геометрия, winding."""

from __future__ import annotations

import json
import struct

import numpy as np

COMPONENTS = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
TYPES = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


class GlbCheckError(Exception):
    pass


def read_glb(path: str) -> tuple[dict, bytes]:
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 20 or data[:4] != b"glTF":
        raise GlbCheckError("не GLB")
    version, length = struct.unpack_from("<II", data, 4)
    if version != 2:
        raise GlbCheckError(f"версия glTF {version} != 2")
    if length != len(data):
        raise GlbCheckError(f"длина в заголовке {length} != фактической {len(data)}")

    offset = 12
    doc = None
    binary = b""
    while offset < len(data):
        clen, ctype = struct.unpack_from("<I4s", data, offset)
        chunk = data[offset + 8 : offset + 8 + clen]
        if ctype == b"JSON":
            doc = json.loads(chunk.decode("utf-8"))
        elif ctype == b"BIN\x00":
            binary = chunk
        offset += 8 + clen
    if doc is None:
        raise GlbCheckError("нет JSON-чанка")
    return doc, binary


def accessor_array(doc: dict, binary: bytes, index: int) -> np.ndarray:
    acc = doc["accessors"][index]
    view = doc["bufferViews"][acc["bufferView"]]
    comp = acc["componentType"]
    ncomp = TYPES[acc["type"]]
    dtype = {
        5126: "<f4",
        5125: "<u4",
        5123: "<u2",
        5121: "<u1",
        5122: "<i2",
        5120: "<i1",
    }[comp]
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    count = acc["count"] * ncomp
    arr = np.frombuffer(binary, dtype=dtype, count=count, offset=start)
    if ncomp > 1:
        arr = arr.reshape(acc["count"], ncomp)
    return arr


def node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4, order="F")
    m = np.eye(4)
    t = node.get("translation") or [0, 0, 0]
    q = node.get("rotation") or [0, 0, 0, 1]
    s = node.get("scale") or [1, 1, 1]
    x, y, z, w = q
    r = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    m[:3, :3] = r * np.array(s)
    m[:3, 3] = t
    return m


def world_matrices(doc: dict) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}

    def walk(idx: int, parent: np.ndarray) -> None:
        m = parent @ node_matrix(doc["nodes"][idx])
        out[idx] = m
        for c in doc["nodes"][idx].get("children", []):
            walk(c, m)

    for scene in doc.get("scenes", []):
        for idx in scene.get("nodes", []):
            walk(idx, np.eye(4))
    return out


def check(path: str, expect_min_y: float | None = None) -> dict:
    doc, binary = read_glb(path)
    problems: list[str] = []

    # min/max обязателен только для POSITION (по спецификации glTF)
    pos_accessors = set()
    for mesh in doc.get("meshes", []):
        for prim in mesh["primitives"]:
            if "POSITION" in prim["attributes"]:
                pos_accessors.add(prim["attributes"]["POSITION"])
    for i in pos_accessors:
        if "min" not in doc["accessors"][i]:
            problems.append(f"POSITION accessor {i} без min/max")

    worlds = world_matrices(doc)
    tris = 0
    verts = 0
    winding_ok = 0
    winding_total = 0
    min_y = None
    max_y = None
    for ni, node in enumerate(doc["nodes"]):
        if "mesh" not in node:
            continue
        w = worlds.get(ni, np.eye(4))
        for prim in doc["meshes"][node["mesh"]]["primitives"]:
            pos = accessor_array(doc, binary, prim["attributes"]["POSITION"])
            idx = accessor_array(doc, binary, prim["indices"]).astype(np.int64)
            if len(idx) % 3:
                problems.append(f"{node['name']}: индексов {len(idx)} не кратно 3")
            if idx.max(initial=0) >= len(pos):
                problems.append(f"{node['name']}: индекс за пределами вершин")
            verts += len(pos)
            tris += len(idx) // 3

            world_pos = pos @ w[:3, :3].T + w[:3, 3]
            if len(world_pos):
                lo = float(world_pos[:, 1].min())
                hi = float(world_pos[:, 1].max())
                min_y = lo if min_y is None else min(min_y, lo)
                max_y = hi if max_y is None else max(max_y, hi)

            if "NORMAL" in prim["attributes"] and len(idx) >= 3:
                nrm = accessor_array(doc, binary, prim["attributes"]["NORMAL"]).astype(np.float64)
                # нормали тоже переводим в мир: иначе сравнение с мировыми гранями врёт
                n3 = w[:3, :3]
                det3 = np.linalg.det(n3)
                if abs(det3) > 1e-9:
                    nrm = nrm @ np.linalg.inv(n3)
                ln = np.linalg.norm(nrm, axis=1, keepdims=True)
                ln[ln == 0] = 1
                nrm = nrm / ln
                tri = idx.reshape(-1, 3)
                a, b, c = world_pos[tri[:, 0]], world_pos[tri[:, 1]], world_pos[tri[:, 2]]
                fn = np.cross(b - a, c - a)
                ln = np.linalg.norm(fn, axis=1, keepdims=True)
                ln[ln == 0] = 1
                fn = fn / ln
                vn = nrm[tri].mean(axis=1)
                vn = vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
                good = (np.einsum("ij,ij->i", fn, vn) > 0)
                winding_ok += int(good.sum())
                winding_total += len(tri)

    det_root = None
    if doc.get("scenes"):
        roots = doc["scenes"][doc.get("scene", 0)].get("nodes", [])
        if roots:
            det_root = float(np.linalg.det(worlds[roots[0]][:3, :3]))

    if expect_min_y is not None and min_y is not None and abs(min_y - expect_min_y) > 0.02:
        problems.append(f"нижняя точка геометрии y={min_y:.4f}, ожидалось {expect_min_y}")

    return {
        "file": path,
        "bytes": len(binary),
        "nodes": len(doc.get("nodes", [])),
        "meshes": len(doc.get("meshes", [])),
        "materials": len(doc.get("materials", [])),
        "textures": len(doc.get("textures", [])),
        "vertices": verts,
        "triangles": tris,
        "min_y": round(min_y, 4) if min_y is not None else None,
        "max_y": round(max_y, 4) if max_y is not None else None,
        "winding_ok": round(winding_ok / winding_total, 4) if winding_total else None,
        "root_det": det_root,
        "problems": problems,
    }