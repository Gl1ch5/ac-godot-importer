#!/usr/bin/env python3
"""Генератор Godot-сцены: рендер всех ТВ-камер трассы по data/cameras.ini.

    python tools/make_camera_scene.py <mod_dir> <out_dir> [--width 1600] [--height 900]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from acimport import inifile as INI

TEMPLATE = """extends Node3D

var cameras = [
{cams}]

func _fix_textures(root: Node) -> void:
	# Рантайм-загрузка GLB не генерирует мип-цепочку: без неё тайлящийся асфальт
	# и сетки превращаются в «телевизионный снег». Строим мипы и включаем AF.
	for mi in root.find_children("*", "MeshInstance3D", true, false):
		var mesh: Mesh = mi.mesh
		if mesh == null:
			continue
		for si in range(mesh.get_surface_count()):
			var mat = mesh.surface_get_material(si)
			if mat is StandardMaterial3D:
				var m2: StandardMaterial3D = mat.duplicate()
				m2.texture_filter = BaseMaterial3D.TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC
				for prop in ["albedo_texture", "normal_texture", "metallic_texture", "roughness_texture", "emission_texture"]:
					var t = m2.get(prop)
					if t is Texture2D:
						var img: Image = t.get_image()
						if img != null and img.get_width() > 1:
							img.generate_mipmaps()
							m2.set(prop, ImageTexture.create_from_image(img))
				mesh.surface_set_material(si, m2)


func _ready():
	var env = WorldEnvironment.new()
	var e = Environment.new()
	var sky = Sky.new()
	var psky = ProceduralSkyMaterial.new()
	psky.sky_top_color = Color(0.20, 0.44, 0.80)
	psky.sky_horizon_color = Color(0.70, 0.80, 0.92)
	psky.ground_bottom_color = Color(0.17, 0.19, 0.20)
	psky.ground_horizon_color = Color(0.64, 0.73, 0.86)
	psky.sun_angle_max = 40.0
	sky.sky_material = psky
	e.background_mode = Environment.BG_SKY
	e.sky = sky
	e.ambient_light_source = Environment.AMBIENT_SOURCE_SKY
	e.ambient_light_sky_contribution = 0.6
	e.ambient_light_energy = 0.55
	e.tonemap_mode = Environment.TONE_MAPPER_ACES
	e.tonemap_exposure = 0.95
	env.environment = e
	add_child(env)

	var sun = DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-45, 45, 0)
	sun.light_color = Color(1.0, 0.98, 0.95)
	sun.light_energy = 1.0
	sun.shadow_enabled = true
	sun.directional_shadow_mode = DirectionalLight3D.SHADOW_PARALLEL_4_SPLITS
	sun.directional_shadow_max_distance = 400.0
	sun.shadow_bias = 0.05
	add_child(sun)

	var vp = get_viewport()
	vp.msaa_3d = Viewport.MSAA_4X
	vp.screen_space_aa = Viewport.SCREEN_SPACE_AA_FXAA
	vp.use_debanding = true

	var gltf = GLTFDocument.new()
	var state = GLTFState.new()
	var err = gltf.append_from_file("res://track.glb", state)
	if err != OK:
		print("ERROR loading glb: ", err)
		get_tree().quit(1)
		return
	var model = gltf.generate_scene(state)
	add_child(model)
	_fix_textures(model)

	var cam = Camera3D.new()
	cam.far = 6000.0
	cam.near = 0.3
	add_child(cam)
	cam.current = true

	DirAccess.make_dir_absolute("res://renders")

	for i in range(cameras.size()):
		var c = cameras[i]
		cam.position = c["pos"]
		cam.look_at(c["target"], Vector3.UP)
		cam.fov = c["fov"]
		for f in range(6):
			await get_tree().process_frame
		var img = vp.get_texture().get_image()
		# SSAA 2x: рендерим в двойном разрешении и уменьшаем — убирает «лесенки» и муар на сетках
		img.resize(img.get_width() / 2, img.get_height() / 2, Image.INTERPOLATE_LANCZOS)
		var fname = "res://renders/cam_%02d_%s.png" % [i + 1, c["id"]]
		img.save_png(fname)
		print("[OK] ", c["title"], " -> ", fname)

	print("ALL RENDERS FINISHED")
	get_tree().quit(0)
"""

SCENE = """[gd_scene load_steps=2 format=3]

[ext_resource type="Script" path="res://render_all_cameras.gd" id="1_script"]

[node name="Root" type="Node3D"]
script = ExtResource("1_script")
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mod_dir")
    ap.add_argument("out_dir")
    args = ap.parse_args()

    mod = Path(args.mod_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ini = INI.load(str(mod / "data" / "cameras.ini"))
    cams = []
    for sec in sorted(ini.sections):
        if not sec.startswith("camera_"):
            continue
        pos = ini.get_list(sec, "POSITION")
        fwd = ini.get_list(sec, "FORWARD")
        up = ini.get_list(sec, "UP")
        if len(pos) < 3 or len(fwd) < 3:
            continue
        name = ini.get(sec, "NAME", sec)
        clean = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower() or sec
        fov = ini.get_float(sec, "MAX_FOV", 10.0)
        # AC TV-камеры телеобъективные (4-30°): для обзорного кадра расширяем
        render_fov = max(35.0, min(60.0, fov * 2.6))
        dist = 60.0
        target = [round(pos[i] + fwd[i] * dist, 3) for i in range(3)]
        cams.append(
            f'\t{{"id": "{clean}", "title": "{name}", '
            f'"pos": Vector3({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), '
            f'"target": Vector3({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}), '
            f'"fov": {render_fov:.1f}}},'
        )

    out.joinpath("render_all_cameras.gd").write_text(
        TEMPLATE.format(cams="\n".join(cams) + "\n"), encoding="utf-8"
    )
    out.joinpath("render_all_cameras.tscn").write_text(SCENE, encoding="utf-8")
    print(f"камер: {len(cams)} -> {out}/render_all_cameras.gd")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())