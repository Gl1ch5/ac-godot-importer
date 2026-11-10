#!/usr/bin/env python3
"""Универсальный импортёр модов Assetto Corsa -> Godot 4 (.glb + json).

    python tools/ac_importer.py <mod_dir|mod.zip> [--id custom_id] [--layout name]
                                [--out DIR] [--max-texture 2048] [--json]

Определение типа:
    CAR   — есть data.acd / data/car.ini / data/suspensions.ini
    TRACK — есть models.ini / models_*.ini / ai/fast_lane.ai

Результат:
    <out>/cars/<id>/car.glb      + meta.json
    <out>/tracks/<id>/track.glb  + meta.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from acimport import gltf as GL          # noqa: E402
from acimport import inifile as INI      # noqa: E402
from acimport import kn5 as K            # noqa: E402
from acimport import scene as SC         # noqa: E402
from acimport import verify as V         # noqa: E402

ARCHIVE_EXT = {".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz"}


# --------------------------------------------------------------------------
# Поиск и распаковка
# --------------------------------------------------------------------------

def unpack_if_archive(path: Path, workdir: Path) -> Path:
    """Распаковать архив (zip/rar/7z) во временную папку."""
    if path.is_dir():
        return path
    if path.suffix.lower() not in ARCHIVE_EXT:
        raise SystemExit(f"Не папка и не архив: {path}")

    dest = workdir / "mod"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[i] распаковка {path.name} -> {dest}")

    ext = path.suffix.lower()
    if ext == ".zip":
        shutil.unpack_archive(str(path), str(dest))
    elif ext in (".tar", ".gz", ".bz2", ".xz"):
        shutil.unpack_archive(str(path), str(dest))
    else:
        # порядок важен: 7z не умеет часть RAR-методов, unrar умеет
        if ext == ".rar" and shutil.which("unrar"):
            subprocess.run(["unrar", "x", "-o+", "-idq", str(path), str(dest) + "/"],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        else:
            tool = next((t for t in ("7z", "7za", "unar", "bsdtar") if shutil.which(t)), None)
            if tool is None:
                raise SystemExit("для .rar/.7z нужен unrar/7z/unar")
            if tool == "unar":
                cmd = [tool, "-o", str(dest), str(path)]
            elif tool == "bsdtar":
                dest.mkdir(parents=True, exist_ok=True)
                cmd = [tool, "-xf", str(path), "-C", str(dest)]
            else:
                cmd = [tool, "x", "-y", f"-o{dest}", str(path)]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return dest


def find_content_root(base: Path) -> Path:
    """Найти папку мода внутри архива: .../content/cars/<id> или сам мод."""
    base = base.resolve()
    if (base / "content").is_dir():
        for kind in ("cars", "tracks"):
            d = base / "content" / kind
            if d.is_dir():
                subs = [p for p in d.iterdir() if p.is_dir()]
                if len(subs) == 1:
                    return subs[0]
                if subs:
                    # берём самую большую по размеру
                    return max(subs, key=lambda p: _dir_size(p))
        return base

    # мод может быть вложен: архив/папка/<car_name>/
    candidates = [base]
    for child in sorted(base.iterdir()):
        if child.is_dir():
            candidates.append(child)
            for sub in child.iterdir():
                if sub.is_dir() and (sub / "data").is_dir():
                    candidates.append(sub)

    best = None
    for c in candidates:
        if (c / "data").is_dir() and ((c / "data" / "car.ini").exists() or (c / "data.acd").exists()):
            return c
        if (c / "models.ini").exists() or list(c.glob("models_*.ini")) or (c / "ai" / "fast_lane.ai").exists():
            best = best or c
    if best:
        return best

    # последняя попытка: папка, где лежит самый большой kn5
    kn5s = sorted(base.rglob("*.kn5"), key=lambda p: p.stat().st_size, reverse=True)
    if kn5s:
        return kn5s[0].parent
    raise SystemExit(f"не понял структуру мода: {base}")


def _dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def detect_kind(mod: Path) -> str:
    data = mod / "data"
    if (mod / "data.acd").exists() or (data / "car.ini").exists() or (data / "suspensions.ini").exists():
        return "car"
    if (mod / "models.ini").exists() or (mod / "data" / "surfaces.ini").exists() or list(mod.glob("models_*.ini")) or (mod / "ai" / "fast_lane.ai").exists():
        return "track"
    kn5_files = list(mod.glob("*.kn5"))
    if kn5_files:
        # Инспекция узлов первого kn5
        try:
            sample = K.load(str(kn5_files[0]))
            if sample.root is not None:
                for n, p, w in K.iter_nodes(sample.root):
                    if n.name in ("WHEEL_LF", "WHEEL_RF", "COCKPIT_HR", "STEER_HR"):
                        return "car"
                    if n.name.startswith("AC_START_") or re.match(r"^[1-9]+[A-Z_]+", n.name):
                        return "track"
        except Exception:
            pass
        names = {p.name.lower() for p in kn5_files}
        if any(k in n for n in names for k in ("track", "road", "circuit", "layout", "phy")):
            return "track"
        if any("body" in n or "lod" in n for n in names):
            return "car"
    raise SystemExit(f"не удалось определить тип мода (CAR/TRACK) в {mod}")


# --------------------------------------------------------------------------
# Выбор моделей
# --------------------------------------------------------------------------

def pick_car_kn5(mod: Path) -> list[Path]:
    """LOD_0 из data/lods.ini, иначе самый большой .kn5."""
    lods = mod / "data" / "lods.ini"
    if lods.exists():
        ini = INI.load(str(lods))
        for section in ("lod_0", "lod0", "car"):
            f = ini.get(section, "file")
            if f:
                p = mod / f
                if p.exists():
                    return [p]
                p2 = mod / Path(f).name
                if p2.exists():
                    return [p2]
    kn5s = sorted(mod.rglob("*.kn5"), key=lambda p: p.stat().st_size, reverse=True)
    if not kn5s:
        raise SystemExit("в моде нет .kn5")
    return [kn5s[0]]


MODEL_POS_RE = re.compile(r"^\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*$")


def _vec3(raw: str | None, default=(0.0, 0.0, 0.0)) -> tuple[float, float, float]:
    if not raw:
        return default
    m = MODEL_POS_RE.match(raw)
    if not m:
        return default
    return tuple(float(v) for v in m.groups())  # type: ignore[return-value]


def _model_matrix(pos: str | None, rot: str | None):
    import numpy as np

    px, py, pz = _vec3(pos)
    rx, ry, rz = _vec3(rot)
    ax, ay, az = np.radians([rx, ry, rz])
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    m = np.eye(4)
    m[:3, :3] = mz @ my @ mx
    m[:3, 3] = (px, py, pz)
    return m


def pick_track_kn5(mod: Path, layout: str | None) -> list[tuple[Path, object]]:
    """models.ini (или models_<layout>.ini) -> [(kn5, матрица)]."""
    ini_path = None
    if layout:
        for name in (f"models_{layout}.ini", f"models-{layout}.ini", f"{layout}.ini"):
            p = mod / name
            if p.exists():
                ini_path = p
                break
    if ini_path is None:
        p = mod / "models.ini"
        if p.exists():
            ini_path = p

    if ini_path is not None:
        ini = INI.load(str(ini_path))
        out = []
        for section in sorted(ini.sections):
            if not section.startswith("model"):
                continue
            f = ini.get(section, "file")
            if not f:
                continue
            p = mod / f
            if not p.exists():
                p = mod / Path(f).name
            if not p.exists():
                continue
            if (ini.get(section, "active") or "1").strip() == "0":
                continue
            out.append((p, _model_matrix(ini.get(section, "position"), ini.get(section, "rotation"))))
        if out:
            return out

    kn5s = sorted(mod.glob("*.kn5"), key=lambda p: p.stat().st_size, reverse=True)
    if not kn5s:
        raise SystemExit("в моде нет .kn5")
    import numpy as np

    return [(p, np.eye(4)) for p in kn5s]


def guess_id(mod: Path) -> str:
    name = mod.name.strip().lower()
    name = re.sub(r"[^a-z0-9_\-]+", "_", name)
    return name.strip("_") or "mod"


# --------------------------------------------------------------------------
# Основные пайплайны
# --------------------------------------------------------------------------

def import_car(mod: Path, car_id: str, out_root: Path, opts: SC.Options) -> dict:
    t0 = time.time()
    files = pick_car_kn5(mod)
    print(f"[i] машина: {mod.name} -> id={car_id}")
    for f in files:
        print(f"    kn5: {f.relative_to(mod)} ({f.stat().st_size / 1e6:.1f} МБ)")

    kn5 = K.load(str(files[0]))
    print(f"[i] kn5 v{kn5.version}: {kn5.stats()}")

    builder = GL.GlbBuilder("ac_importer/car")
    info = SC.build_car(kn5, builder, opts, mod_dir=mod)

    dest = out_root / "cars" / car_id
    dest.mkdir(parents=True, exist_ok=True)
    glb_path = dest / "car.glb"
    size = builder.save(str(glb_path))

    report = V.check(str(glb_path))
    meta = {
        "id": car_id,
        "name": mod.name,
        "kind": "car",
        "glb": glb_path.name,
        "source_kn5": [f.name for f in files],
        "wheels": info["wheels"],
        "wheel_layout": SC.wheel_layout_report(info["wheels"]),
        "steer_node": info["steer"],
        "ground_offset": info.get("min_y_before_offset"),
        "materials": _material_flags(builder),
        "stats": {**report, "glb_bytes": size},
        "texture_errors": info.get("texture_errors", []),
        "skipped_nodes": info["stats"]["skipped"][:50],
        "imported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seconds": round(time.time() - t0, 2),
    }
    (dest / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] {glb_path} — {size / 1e6:.2f} МБ, {report['triangles']} тр., колеса: {meta['wheel_layout']}")
    return meta


def import_track(mod: Path, track_id: str, out_root: Path, opts: SC.Options, layout: str | None) -> dict:
    t0 = time.time()
    entries = pick_track_kn5(mod, layout)
    print(f"[i] трасса: {mod.name} -> id={track_id} (моделей: {len(entries)})")

    kn5_entries = []
    total = 0
    for p, m in entries:
        kn5 = K.load(str(p))
        kn5_entries.append((kn5, m))
        total += p.stat().st_size
        print(f"    kn5: {p.name} ({p.stat().st_size / 1e6:.1f} МБ) v{kn5.version}: {kn5.stats()}")

    builder = GL.GlbBuilder("ac_importer/track")
    info = SC.build_track(kn5_entries, builder, opts, mod_dir=mod)

    dest = out_root / "tracks" / track_id
    dest.mkdir(parents=True, exist_ok=True)
    glb_path = dest / "track.glb"
    size = builder.save(str(glb_path))
    report = V.check(str(glb_path))

    meta = {
        "id": track_id,
        "name": mod.name,
        "kind": "track",
        "layout": layout,
        "glb": glb_path.name,
        "source_kn5": [p.name for p, _ in entries],
        "source_bytes": total,
        "start": info["start"],
        "pit": info["pit"],
        "surfaces": info["surfaces"],
        "collision_triangles": sum(info["surfaces"].values()),
        "visual_nodes": info["mesh_nodes"],
        "materials": _material_flags(builder),
        "stats": {**report, "glb_bytes": size},
        "texture_errors": info.get("texture_errors", [])[:20],
        "imported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seconds": round(time.time() - t0, 2),
    }
    (dest / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    ratio = (size / total * 100) if total else 0
    print(f"[ok] {glb_path} — {size / 1e6:.2f} МБ ({ratio:.0f}% от исходных {total / 1e6:.0f} МБ), "
          f"узлов {report['nodes']}, треугольников {report['triangles']}, "
          f"коллизия {meta['collision_triangles']} тр.")
    if report["problems"]:
        print("[!] проблемы GLB: " + "; ".join(report["problems"][:5]))
    return meta


def _material_flags(builder: GL.GlbBuilder) -> dict:
    out = {}
    for m in builder.materials:
        out[m["name"]] = {
            "alpha": m.get("alphaMode", "OPAQUE"),
            "double_sided": bool(m.get("doubleSided")),
            "clearcoat": "KHR_materials_clearcoat" in (m.get("extensions") or {}),
            "shader": (m.get("extras") or {}).get("ac_shader", ""),
        }
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Assetto Corsa -> Godot 4 (.glb) импортёр")
    ap.add_argument("path", help="папка или архив мода Assetto Corsa")
    ap.add_argument("--id", dest="mod_id", help="id в проекте (по умолчанию имя папки)")
    ap.add_argument("--layout", help="layout трассы (models_<layout>.ini)")
    ap.add_argument("--out", default="sim/assets", help="корень ассетов проекта (по умолчанию sim/assets)")
    ap.add_argument("--max-texture", type=int, default=2048, help="макс. размер текстуры")
    ap.add_argument("--keep-cockpit", action="store_true", help="не выкидывать COCKPIT_LR")
    ap.add_argument("--keep-blur", action="store_true", help="не выкидывать BLUR_* колёса")
    ap.add_argument("--json", action="store_true", help="только JSON итога")
    ap.add_argument("--keep-temp", action="store_true", help="не удалять распакованное")
    ap.add_argument("--no-tangents", action="store_true", help="не писать TANGENT (Godot сгенерирует сам)")
    ap.add_argument("--keep-decals", action="store_true", help="не выкидывать копланарные декали (накат, трещины)")
    args = ap.parse_args(argv)

    path = Path(args.path).expanduser()
    if not path.exists():
        print(f"нет такого пути: {path}", file=sys.stderr)
        return 2

    out_root = Path(args.out).expanduser()
    opts = SC.Options(max_texture=args.max_texture, keep_cockpit=args.keep_cockpit,
                      keep_blur=args.keep_blur, no_tangents=args.no_tangents,
                      drop_decals=not args.keep_decals)

    workdir = Path(tempfile.mkdtemp(prefix="ac_import_"))
    try:
        mod = find_content_root(unpack_if_archive(path, workdir))
        kind = detect_kind(mod)
        mod_id = args.mod_id or guess_id(mod)
        print(f"[i] тип: {kind.upper()}, папка: {mod}")

        if kind == "car":
            meta = import_car(mod, mod_id, out_root, opts)
        else:
            meta = import_track(mod, mod_id, out_root, opts, args.layout)

        if args.json:
            print(json.dumps(meta, ensure_ascii=False, indent=2))
        return 0 if not meta["stats"]["problems"] else 1
    finally:
        if args.keep_temp:
            print(f"[i] временные файлы: {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())