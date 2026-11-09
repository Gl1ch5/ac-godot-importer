#!/usr/bin/env python3
"""Автотесты конвертера: разбор KN5, математика осей, материалы, GLB, CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from acimport import geometry as G     # noqa: E402
from acimport import gltf as GL        # noqa: E402
from acimport import kn5 as K          # noqa: E402
from acimport import materials as M    # noqa: E402
from acimport import scene as SC       # noqa: E402
from acimport import textures as T     # noqa: E402
from acimport import verify as V       # noqa: E402
from tests import fixture as F         # noqa: E402

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        PASS.append(name)
        print(f"  ok   {name}")
    else:
        FAIL.append(f"{name} {detail}")
        print(f"  FAIL {name} {detail}")


def approx(a, b, eps=1e-3) -> bool:
    return abs(float(a) - float(b)) <= eps


# --------------------------------------------------------------------------

def test_parser():
    print("\n[1] Парсер KN5")
    data = F.make_car_kn5()
    with tempfile.NamedTemporaryFile(suffix=".kn5", delete=False) as f:
        f.write(data)
        path = f.name
    kn5 = K.load(path)
    check("magic/version", kn5.version == 5, str(kn5.version))
    check("текстур 3", len(kn5.textures) == 3, str(len(kn5.textures)))
    check("материалов 3", len(kn5.materials) == 3, str(len(kn5.materials)))
    check("шейдер body_paint", kn5.materials[0].shader == "ksPerPixelMultiMap", kn5.materials[0].shader)
    check("txNormal найден", kn5.materials[0].texture("txNormal") == "body_normal.dds")
    check("alpha_tested у дерева", kn5.materials[2].alpha_tested is True)
    check("blend у стекла", kn5.materials[1].blend_mode == 1)

    root = kn5.root
    check("корень Root", root.name == "Root" and root.cls == 1)
    wheels = {n.name for n in root.walk()}
    check("все WHEEL_*", {"WHEEL_LF", "WHEEL_RF", "WHEEL_LR", "WHEEL_RR"} <= wheels)

    lf = root.find("WHEEL_LF")
    w = SC._node_world(root, "WHEEL_LF")
    check("WHEEL_LF x=-0.82", approx(w[0, 3], -0.82), f"{w[0, 3]}")
    check("WHEEL_LF z=-1.35", approx(w[2, 3], -1.35), f"{w[2, 3]}")

    stats = kn5.stats()
    check("треугольников > 0", stats["triangles"] > 0, str(stats))
    check("не скин-меш", lf.children[0].cls == K.NODE_MESH)
    os.unlink(path)


def test_geometry():
    print("\n[2] Геометрия: winding и нормали")
    pos, nrm, uv, idx = F.box((1, 1, 1))
    check("куб: обход согласован с нормалями", G.vertex_normal_sanity(pos.astype(np.float64), nrm.astype(np.float64), idx.astype(np.int64)) == 1.0)

    # отражение по X должно перевернуть обход, bake() это компенсирует
    mir = np.diag([-1.0, 1.0, 1.0, 1.0])
    p2 = K.transform_points(mir, pos.astype(np.float64))
    bad = G.vertex_normal_sanity(p2, nrm.astype(np.float64), idx.astype(np.int64))
    check("зеркало без фикса ломает winding", bad < 0.8, f"{bad}")

    node = K.Node(name="m", cls=K.NODE_MESH)
    node.vertices = np.concatenate([pos, nrm, uv, np.zeros((len(pos), 3), np.float32)], axis=1).astype(np.float32)
    node.indices = idx
    p3, n3, u3, i3 = G.bake(node, mir)
    fixed = G.vertex_normal_sanity(p3, n3, i3.astype(np.int64))
    check("bake() восстанавливает winding", fixed == 1.0, f"{fixed}")

    tang = G.compute_tangents(pos.astype(np.float64), nrm.astype(np.float64), uv.astype(np.float64), idx.astype(np.int64))
    check("тангенты посчитаны", tang is not None and tang.shape == (len(pos), 4))

    tilted = G.fix_tree_normals(np.tile([0.0, -1.0, 0.0], (4, 1)))
    check("ksTree: нормали кверху", (tilted[:, 1] > 0.5).all(), str(tilted[0]))


def test_materials():
    print("\n[3] Материалы -> PBR")
    data = F.make_car_kn5()
    with tempfile.NamedTemporaryFile(suffix=".kn5", delete=False) as f:
        f.write(data)
        path = f.name
    kn5 = K.load(path)
    cache = M.TextureCache(kn5)

    body = M.convert(kn5.materials[0], cache)
    glass = M.convert(kn5.materials[1], cache)
    tree = M.convert(kn5.materials[2], cache)

    check("кузов OPAQUE", body.alpha_mode == "OPAQUE", body.alpha_mode)
    check("кузов без альфы в baseColor", body.base_color_texture.mode == "RGB", body.base_color_texture.mode)
    check("стекло BLEND", glass.alpha_mode == "BLEND", glass.alpha_mode)
    check("дерево MASK 0.4", tree.alpha_mode == "MASK" and approx(tree.alpha_cutoff, 0.4))
    check("дерево doubleSided", tree.double_sided is True)
    check("кузов — краска (clearcoat)", body.is_paint is True)
    check("metallicRoughness собрана", body.metallic_texture is not None)

    # нормаль: G должен инвертироваться (64 -> 191)
    nimg = np.array(body.normal_texture)
    check("normal map: G инвертирован", abs(int(nimg[0, 0, 1]) - (255 - 64)) <= 1, str(nimg[0, 0]))

    # txMaps: gloss=200 -> rough=55, metal=128
    mr = np.array(body.metallic_texture)
    check("metallicRoughness: roughness = 255-gloss", abs(int(mr[0, 0, 1]) - 55) <= 1, str(mr[0, 0]))
    check("metallicRoughness: metallic = B", abs(int(mr[0, 0, 2]) - 128) <= 2, str(mr[0, 0]))
    os.unlink(path)


def test_car_glb():
    print("\n[4] Машина -> GLB")
    data = F.make_car_kn5()
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "car.kn5")
    with open(path, "wb") as f:
        f.write(data)
    kn5 = K.load(path)
    b = GL.GlbBuilder("test")
    info = SC.build_car(kn5, b, SC.Options())
    out = os.path.join(tmp, "car.glb")
    size = b.save(out)

    rep = V.check(out)
    check("GLB валиден", not rep["problems"], str(rep["problems"]))
    check("определитель корня +1", approx(rep["root_det"], 1.0), f"{rep['root_det']}")
    check("winding 100%", rep["winding_ok"] == 1.0, f"{rep['winding_ok']}")
    check("посадка y=0", approx(rep["min_y"], 0.0, 0.01), f"{rep['min_y']}")

    layout = SC.wheel_layout_report(info["wheels"])
    check("колёса в квадрантах Godot", all(v == "ok" for v in layout.values()), str(layout))
    check("руль найден", info["steer"] == "STEER_HR", str(info["steer"]))

    names = {n.get("name") for n in b.nodes}
    check("BLUR_* выкинут", "BLUR_LF" not in names)
    check("COCKPIT_LR выкинут", "COCKPIT_LR" not in names)
    check("колесо сохранено", "WHEEL_LF" in names)

    mats = {m["name"]: m for m in b.materials}
    check("alphaMode кузова в GLB", mats["body_paint"]["alphaMode"] == "OPAQUE")
    check("clearcoat в GLB", "KHR_materials_clearcoat" in mats["body_paint"].get("extensions", {}))
    check("лёгкий файл (<2 МБ)", size < 2_000_000, f"{size}")
    print(f"     (GLB {size/1024:.0f} КБ, {rep['triangles']} тр., узлов {rep['nodes']})")
    return out


def test_track_glb():
    print("\n[5] Трасса -> GLB (оси 1:1, батчинг, коллизия)")
    data = F.make_track_kn5()
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "track.kn5")
    with open(path, "wb") as f:
        f.write(data)
    kn5 = K.load(path)

    b = GL.GlbBuilder("test")
    info = SC.build_track([kn5], b, SC.Options())
    out = os.path.join(tmp, "track.glb")
    b.save(out)
    rep = V.check(out)

    doc, binary = V.read_glb(out)
    check("GLB валиден", not rep["problems"], str(rep["problems"]))

    # 1:1 по осям: пол трассы в KN5 тоже y=0, x ∈ [-6,6]
    check("min_y = -0.2 (grass)", approx(rep["min_y"], -0.2, 0.01), f"{rep['min_y']}")
    check("max_y = 1.0 (wall)", approx(rep["max_y"], 1.0, 0.01), f"{rep['max_y']}")

    road_node = next((n for n in b.nodes if n.get("name") == "vis_road_mat"), None)
    check("батчинг: один узел на материал", road_node is not None)
    vnames = [n.get("name") for n in b.nodes if n.get("name", "").startswith("vis_")]
    check("узлов визуала мало", len(vnames) <= 3, str(vnames))

    pos = None
    if road_node:
        prim = b.meshes[road_node["mesh"]]["primitives"][0]
        idx = b.accessors[prim["indices"]]
        pos = V.accessor_array(doc, binary, prim["attributes"]["POSITION"])
    check("дорога 12 x 60 (без инверсии осей)",
          pos is not None and approx(np.ptp(pos[:, 0]), 12.0, 0.01) and approx(np.ptp(pos[:, 2]), 60.0, 0.01),
          "" if pos is None else f"x={np.ptp(pos[:,0]):.2f} z={np.ptp(pos[:,2]):.2f}")

    check("winding 100%", rep["winding_ok"] == 1.0, f"{rep['winding_ok']}")
    check("стена ушла в коллизию", "wall" in info["surfaces"], str(info["surfaces"]))
    check("визуал: дорога+трава", set(info["surfaces"].keys()) == {"wall"}, str(info["surfaces"]))
    check("спавн AC_START_0", info["start"] is not None and approx(info["start"][2], 20.0), str(info["start"]))
    return out


def test_cli():
    print("\n[6] CLI end-to-end")
    tmp = tempfile.mkdtemp()
    car_dir = os.path.join(tmp, "content", "cars", "test_car")
    os.makedirs(car_dir)
    F.write_car_mod(car_dir)

    out_root = os.path.join(tmp, "assets")
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "ac_importer.py"), car_dir,
                        "--out", out_root, "--json"], capture_output=True, text=True)
    check("CLI код 0", r.returncode == 0, r.stderr[-400:])
    meta_path = os.path.join(out_root, "cars", "test_car", "meta.json")
    check("meta.json создан", os.path.exists(meta_path))
    if os.path.exists(meta_path):
        meta = json.loads(open(meta_path, encoding="utf-8").read())
        check("тип car", meta["kind"] == "car")
        check("колёс 4", len(meta["wheels"]) == 4, str(meta["wheels"]))
        check("нет проблем GLB", not meta["stats"]["problems"], str(meta["stats"]["problems"]))

    track_dir = os.path.join(tmp, "content", "tracks", "test_track")
    os.makedirs(track_dir)
    F.write_track_mod(track_dir)
    r2 = subprocess.run([sys.executable, str(ROOT / "tools" / "ac_importer.py"), track_dir,
                         "--out", out_root], capture_output=True, text=True)
    check("CLI трассы код 0", r2.returncode == 0, r2.stderr[-400:])
    tmeta = os.path.join(out_root, "tracks", "test_track", "meta.json")
    check("track meta.json", os.path.exists(tmeta))
    if os.path.exists(tmeta):
        m = json.loads(open(tmeta, encoding="utf-8").read())
        check("тип track", m["kind"] == "track")
        src = m["source_bytes"]
        glb = m["stats"]["glb_bytes"]
        check("glb меньше исходника", glb < src, f"{glb} vs {src}")


def main():
    test_parser()
    test_geometry()
    test_materials()
    test_car_glb()
    test_track_glb()
    test_cli()
    print(f"\nИтого: {len(PASS)} ok, {len(FAIL)} fail")
    if FAIL:
        for f in FAIL:
            print("  FAIL:", f)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())