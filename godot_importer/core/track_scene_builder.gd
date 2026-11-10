class_name TrackSceneBuilder
extends RefCounted

const ACMaterialFactory = preload("res://materials/ac_material_factory.gd")
const ACSpawnExtractor = preload("res://spec/ac_spawn_extractor.gd")
const ACCollisionBuilder = preload("res://physics/ac_collision_builder.gd")

var mat_factory: RefCounted

func _init(p_mat_factory) -> void:
	mat_factory = p_mat_factory

func setup_track_scene(model_root: Node3D, ac_mat_db: Dictionary, track_spec_data: Dictionary) -> void:
	# 1. TrackSpec (спавны, пит-боксы, тайминг)
	ACSpawnExtractor.populate_track_spec(model_root, track_spec_data)

	# 2. Обход мешей: фильтрация служебных дамми, маппинг материалов, авто-физика
	var processed_mats = {}
	for mi in model_root.find_children("*", "MeshInstance3D", true, false):
		# Меши коллизии surf_* в рендере скрываем
		if mi.name.begins_with("surf_") or mi.name == "Collision":
			mi.visible = false
			continue

		# Скрываем ТОЛЬКО служебные маркеры
		if ACSpawnExtractor.is_dummy_marker(mi.name):
			mi.visible = false
			continue
		else:
			mi.visible = true

		var mesh: Mesh = mi.mesh
		if mesh == null:
			continue

		for si in range(mesh.get_surface_count()):
			var mat = mesh.surface_get_material(si)
			if mat == null:
				continue
			var mat_id = mat.get_instance_id()
			if processed_mats.has(mat_id):
				mesh.surface_set_material(si, processed_mats[mat_id])
				continue

			var orig_name = mat.resource_name.to_lower()
			var info: Dictionary = ac_mat_db.get(orig_name, {
				"name": mat.resource_name,
				"shader": "ksPerPixel",
				"blend_mode": 0,
				"alpha_tested": false,
				"properties": {},
				"textures": {"txDiffuse": orig_name}
			})

			var new_mat = mat_factory.create_material(info)
			processed_mats[mat_id] = new_mat
			mesh.surface_set_material(si, new_mat)

		# Авто-физика для Kunos-мешей
		if ACCollisionBuilder.is_physical_mesh(mi.name):
			pass
