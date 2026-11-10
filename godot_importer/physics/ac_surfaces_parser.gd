class_name ACSurfacesParser
extends RefCounted

static func parse_surfaces(ini_text: String) -> Dictionary:
	var surfaces = {}
	var cur_sec = ""
	for line in ini_text.split("\n"):
		var s = line.strip_edges()
		if s.begins_with(";") or s.begins_with("//") or s.is_empty():
			continue
		if s.begins_with("[") and s.ends_with("]"):
			cur_sec = s.substr(1, s.length() - 2).to_upper()
			surfaces[cur_sec] = {}
		elif cur_sec != "" and "=" in s:
			var parts = s.split("=", false, 2)
			var key = parts[0].strip_edges().to_upper()
			var val = parts[1].strip_edges()
			surfaces[cur_sec][key] = val
	return surfaces
