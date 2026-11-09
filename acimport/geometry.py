"""Геометрия: запекание трансформаций, тангенты, фиксы нормалей, батчинг."""

from __future__ import annotations

import numpy as np

from . import kn5 as K

EPS = 1e-9


def positions_of(node: K.Node) -> np.ndarray:
    return np.asarray(node.vertices[:, 0:3], dtype=np.float64)


def normals_of(node: K.Node) -> np.ndarray:
    return np.asarray(node.vertices[:, 3:6], dtype=np.float64)


def uvs_of(node: K.Node) -> np.ndarray:
    uv = np.asarray(node.vertices[:, 6:8], dtype=np.float64).copy()
    return uv


def tangents_of(node: K.Node) -> np.ndarray:
    return np.asarray(node.vertices[:, 8:11], dtype=np.float64)


def bake(node: K.Node, world: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Локальная геометрия -> мировые координаты (column-vector конвенция)."""
    pos = K.transform_points(world, positions_of(node))
    nrm = K.transform_normals(world, normals_of(node))
    uv = uvs_of(node)
    idx = np.asarray(node.indices, dtype=np.int64).reshape(-1)

    det = np.linalg.det(world[:3, :3])
    if det < 0:  # отражение: обход нужно вернуть, иначе нормали «внутрь»
        idx = idx.reshape(-1, 3)[:, [0, 2, 1]].reshape(-1)
    return pos, nrm, uv, idx


def compute_tangents(pos: np.ndarray, nrm: np.ndarray, uv: np.ndarray, idx: np.ndarray) -> np.ndarray | None:
    """MikkTSpace-подобный расчёт: T из UV-градиентов, w — ориентация."""
    if len(pos) == 0 or len(idx) < 3:
        return None
    tri = idx.reshape(-1, 3)
    p0, p1, p2 = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    t0, t1, t2 = uv[tri[:, 0]], uv[tri[:, 1]], uv[tri[:, 2]]

    dp1 = p1 - p0
    dp2 = p2 - p0
    duv1 = t1 - t0
    duv2 = t2 - t0

    denom = duv1[:, 0] * duv2[:, 1] - duv2[:, 0] * duv1[:, 1]
    ok = np.abs(denom) > 1e-12
    if not ok.any():
        return None
    r = np.where(ok, 1.0 / np.where(ok, denom, 1.0), 0.0)
    tan = (dp1 * duv2[:, 1:2] - dp2 * duv1[:, 1:2]) * r[:, None]
    bit = (dp2 * duv1[:, 0:1] - dp1 * duv2[:, 0:1]) * r[:, None]

    out = np.zeros((len(pos), 3), dtype=np.float64)
    outb = np.zeros((len(pos), 3), dtype=np.float64)
    for k in range(3):
        np.add.at(out, tri[:, k], tan)
        np.add.at(outb, tri[:, k], bit)

    n = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), EPS)
    t = out - n * np.einsum("ij,ij->i", n, out)[:, None]
    tl = np.linalg.norm(t, axis=1, keepdims=True)
    bad = (tl[:, 0] < 1e-8)
    if bad.any():
        # вырожденные UV: берём любую ортогональ к нормали
        axis = np.where(np.abs(n[bad, 1:2]) < 0.9, np.array([[0.0, 1.0, 0.0]]), np.array([[0.0, 0.0, 1.0]]))
        fallback = np.cross(axis, n[bad])
        t[bad] = fallback
        tl = np.linalg.norm(t, axis=1, keepdims=True)
    t = t / np.maximum(tl, EPS)

    w = np.sign(np.einsum("ij,ij->i", np.cross(n, t), outb))
    w[w == 0] = 1.0
    return np.concatenate([t, w[:, None]], axis=1).astype(np.float32)


def fix_tree_normals(nrm: np.ndarray, up_weight: float = 0.75) -> np.ndarray:
    """ksTree: N = normalize(lerp(N, up, 0.75)) — лечит чёрные деревья."""
    up = np.zeros_like(nrm)
    up[:, 1] = 1.0
    mixed = nrm * (1.0 - up_weight) + up * up_weight
    ln = np.linalg.norm(mixed, axis=1, keepdims=True)
    ln[ln < EPS] = 1.0
    return mixed / ln


def merge_arrays(parts: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]):
    """Склеить (pos, nrm, uv, idx) в один буфер со сдвигом индексов."""
    if not parts:
        return None
    pos, nrm, uv, idx = [], [], [], []
    base = 0
    has_uv = all(p[2] is not None for p in parts)
    for p, n, u, i in parts:
        pos.append(p)
        nrm.append(n)
        if has_uv:
            uv.append(u)
        idx.append(np.asarray(i, dtype=np.int64) + base)
        base += len(p)
    out_pos = np.concatenate(pos, axis=0)
    out_nrm = np.concatenate(nrm, axis=0)
    out_uv = np.concatenate(uv, axis=0) if has_uv else None
    out_idx = np.concatenate(idx, axis=0)
    return out_pos, out_nrm, out_uv, out_idx


def mesh_bounds(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return pos.min(axis=0), pos.max(axis=0)


def vertex_normal_sanity(pos: np.ndarray, nrm: np.ndarray, idx: np.ndarray) -> float:
    """Доля треугольников, чья геометрическая нормаль смотрит в сторону вершинных.

    1.0 — всё отлично; ~0.0 — обход вывернут (то, чем болел старый конвертер).
    """
    if len(idx) < 3:
        return 1.0
    tri = idx.reshape(-1, 3)
    a, b, c = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    fn = np.cross(b - a, c - a)
    ln = np.linalg.norm(fn, axis=1, keepdims=True)
    ln[ln < EPS] = 1.0
    fn = fn / ln
    vn = nrm[tri].mean(axis=1)
    vn = vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), EPS)
    return float((np.einsum("ij,ij->i", fn, vn) > 0).mean())