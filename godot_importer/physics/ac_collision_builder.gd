class_name ACCollisionBuilder
extends RefCounted

static var KUNOS_PHYS_REGEX = RegEx.create_from_string("^[1-9]+[A-Z_]+")

static func is_physical_mesh(mesh_name: String) -> bool:
	return KUNOS_PHYS_REGEX.search(mesh_name) != null

static func build_trimesh_collision(mesh_instance: MeshInstance3D, surface_name: String) -> StaticBody3D:
	var sb = StaticBody3D.new()
	sb.name = "Col_" + mesh_instance.name
	var shape = ConcavePolygonShape3D.new()
	shape.set_faces(mesh_instance.mesh.get_faces())
	var cs = CollisionShape3D.new()
	cs.shape = shape
	sb.add_child(cs)
	sb.set_meta("ac_surface", surface_name)
	return sb
