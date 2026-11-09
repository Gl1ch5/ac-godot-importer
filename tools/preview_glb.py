#!/usr/bin/env python3
"""Быстрый софтверный превью-рендер GLB (z-buffer + ламберт), без GPU.

    python tools/preview_glb.py model.glb out.png [--views 3] [--size 900]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from acimport import verify as V


def _avg_color(doc, binary, tex_index) -> np.ndarray | None:
    try:
        src = doc["textures"][tex_index]["source"]
        img = doc["images"][src]
        view = doc["bufferViews"][img["bufferView"]]
        start = view.get("byteOffset", 0)
        raw = binary[start : start + view["byteLength"]]
        import io

        from PIL import Image as _I

        im = _I.open(io.BytesIO(raw)).convert("RGB")
        im.thumbnail((32, 32))
        arr = np.asarray(im, dtype=np.float64) / 255.0
        return arr.reshape(-1, 3).mean(axis=0)
    except Exception:
        return None


def gather(doc, binary):
    worlds = V.world_matrices(doc)
    tris = []
    cols = []
    for ni, node in enumerate(doc.get("nodes", [])):
        if "mesh" not in node:
            continue
        w = worlds.get(ni, np.eye(4))
        for prim in doc["meshes"][node["mesh"]]["primitives"]:
            pos = V.accessor_array(doc, binary, prim["attributes"]["POSITION"]).astype(np.float64)
            idx = V.accessor_array(doc, binary, prim["indices"]).astype(np.int64)
            mat = doc.get("materials", [{}])[prim.get("material", 0)] if doc.get("materials") else {}
            pbr = mat.get("pbrMetallicRoughness", {}) or {}
            factor = np.array(pbr.get("baseColorFactor", [0.8, 0.8, 0.8, 1])[:3], dtype=np.float64)
            albedo = None
            if "baseColorTexture" in pbr:
                albedo = _avg_color(doc, binary, pbr["baseColorTexture"]["index"])
            if albedo is None:
                albedo = factor
            else:
                albedo = np.clip(albedo * factor, 0.0, 1.0)
            pw = pos @ w[:3, :3].T + w[:3, 3]
            tris.append(pw[idx.reshape(-1, 3)])
            cols.append(np.tile(albedo, (len(idx) // 3, 1)))
    if not tris:
        raise SystemExit("нет геометрии")
    return np.concatenate(tris), np.concatenate(cols)


def render(tris, cols, size=900, yaw=35.0, pitch=14.0, bg=(0.09, 0.10, 0.12)):
    lo = tris.reshape(-1, 3).min(axis=0)
    hi = tris.reshape(-1, 3).max(axis=0)
    ctr = (lo + hi) * 0.5
    scale = 1.0 / max(hi - lo) * 2.0

    ay, ax = np.radians(yaw), np.radians(pitch)
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    rot = rx @ ry

    v = (tris.reshape(-1, 3) - ctr) @ rot.T * scale
    v = v.reshape(-1, 3, 3)
    depth = v[:, :, 2]
    # простая камера-перспектива
    focal = 3.2
    z = depth + 4.0
    sx = (v[:, :, 0] / z * focal * 0.5 + 0.5) * size
    sy = (1.0 - (v[:, :, 1] / z * focal * 0.5 + 0.5)) * size

    img = np.zeros((size, size, 3), dtype=np.float64)
    img[:] = bg
    zbuf = np.full((size, size), np.inf)

    # освещение
    light = np.array([0.45, 0.8, 0.4])
    light /= np.linalg.norm(light)

    a = np.stack([sx[:, 0], sy[:, 0]], 1)
    b = np.stack([sx[:, 1], sy[:, 1]], 1)
    c = np.stack([sx[:, 2], sy[:, 2]], 1)
    area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    keep = np.abs(area) > 1e-9
    # экранная ось Y перевёрнута, поэтому лицевые (CCW снаружи) дают area > 0
    keep &= area > 0
    if not keep.any():
        keep = np.abs(area) > 1e-9

    e1 = v[:, 1] - v[:, 0]
    e2 = v[:, 2] - v[:, 0]
    fn = np.cross(e1, e2)
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    shade = 0.28 + 0.72 * np.clip(fn @ light, 0.0, 1.0)

    order = np.argsort(-(z.mean(axis=1)))
    for i in order:
        if not keep[i]:
            continue
        x0, x1, x2 = sx[i]
        y0, y1, y2 = sy[i]
        xs = int(max(0, np.floor(min(x0, x1, x2))))
        xe = int(min(size - 1, np.ceil(max(x0, x1, x2))))
        ys = int(max(0, np.floor(min(y0, y1, y2))))
        ye = int(min(size - 1, np.ceil(max(y0, y1, y2))))
        if xe < xs or ye < ys:
            continue
        px, py = np.meshgrid(np.arange(xs, xe + 1) + 0.5, np.arange(ys, ye + 1) + 0.5)
        d = area[i]
        w0 = ((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)) / d
        w1 = ((x2 - px) * (y0 - py) - (x0 - px) * (y2 - py)) / d
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        zi = w0 * depth[i, 0] + w1 * depth[i, 1] + w2 * depth[i, 2]
        sub = zbuf[ys : ye + 1, xs : xe + 1]
        upd = inside & (zi < sub)
        if not upd.any():
            continue
        sub[upd] = zi[upd]
        img[ys : ye + 1, xs : xe + 1][upd] = np.clip(cols[i] * shade[i], 0, 1)

    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    path, out = sys.argv[1], sys.argv[2]
    # Виды сверху (pitch = -22 градуса — смотрим сверху вниз на кузов/капот)
    views = [(35.0, -22.0), (135.0, -22.0), (0.0, -85.0)]  # 3/4 спереди, 3/4 сзади, строго сверху
    doc, binary = V.read_glb(path)
    tris, cols = gather(doc, binary)
    imgs = [render(tris, cols, yaw=y, pitch=p) for y, p in views]
    w, h = imgs[0].size
    sheet = Image.new("RGB", (w * 3, h), (0, 0, 0))
    for i, im in enumerate(imgs):
        sheet.paste(im, (i * w, 0))
    sheet.save(out)
    print(f"{out}: {len(tris)} тр., {w*3}x{h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())