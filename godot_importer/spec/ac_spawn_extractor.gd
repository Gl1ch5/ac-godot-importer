class_name ACSpawnExtractor
extends RefCounted

static var DUMMY_PREFIXES = [
	"AC_START_", "AC_PIT_", "AC_HOTLAP_START_", "AC_TIME_", "AC_AB_", "AC_AUDIO_"
]

static func is_dummy_marker(node_name: String) -> bool:
	for p in DUMMY_PREFIXES:
		if node_name.begins_with(p):
			return true
	return false

static func populate_track_spec(root: Node3D, track_spec_data: Dictionary) -> Node3D:
	var spec = Node3D.new()
	spec.name = "TrackSpec"
	root.add_child(spec)

	var start_grid = Node3D.new()
	start_grid.name = "StartGrid"
	spec.add_child(start_grid)

	var pit_boxes = Node3D.new()
	pit_boxes.name = "PitBoxes"
	spec.add_child(pit_boxes)

	var timing = Node3D.new()
	timing.name = "TimingGates"
	spec.add_child(timing)

	for id in track_spec_data.get("start_grid", {}).keys():
		var pos = track_spec_data["start_grid"][id]
		var m = Marker3D.new()
		m.name = id
		m.position = Vector3(pos[0], pos[1], pos[2])
		start_grid.add_child(m)

	for id in track_spec_data.get("pit_boxes", {}).keys():
		var pos = track_spec_data["pit_boxes"][id]
		var m = Marker3D.new()
		m.name = id
		m.position = Vector3(pos[0], pos[1], pos[2])
		pit_boxes.add_child(m)

	for id in track_spec_data.get("timing_gates", {}).keys():
		var pos = track_spec_data["timing_gates"][id]
		var m = Marker3D.new()
		m.name = id
		m.position = Vector3(pos[0], pos[1], pos[2])
		timing.add_child(m)

	return spec
