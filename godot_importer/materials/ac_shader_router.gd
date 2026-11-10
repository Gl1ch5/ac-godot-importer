class_name ACShaderRouter
extends RefCounted

enum ShaderFamily {
	MULTILAYER,
	MULTIMAP,
	TREE_FOLIAGE,
	ALPHA_CUTOUT,
	DECAL_ALPHA,
	OPAQUE_PBR
}

const DECAL_KEYWORDS = ["groove", "skid", "decal", "mark", "line_white", "line_yellow", "rubber", "stripe", "glass", "window"]
const TREE_KEYWORDS = ["tree", "bush", "foliage", "leaf", "leaves", "hedgerow", "pine", "oak", "forest", "vegetation", "grass_blades", "weed", "plant", "branch"]

static func classify(shader_name: String, blend_mode: int, alpha_tested: bool, mat_name: String = "", tex_slots: Dictionary = {}) -> ShaderFamily:
	var s = shader_name.to_lower()
	var m = mat_name.to_lower()

	# 1. МУЛЬТИСЛОЙНЫЙ АСФАЛЬТ И ТРАВА: если есть txMask и txDetailR (или имя шейдера ksMultilayer*)
	if s.begins_with("ksmultilayer") or s.begins_with("stmultilayer") or (tex_slots.has("txMask") and tex_slots.has("txDetailR")):
		return ShaderFamily.MULTILAYER

	# 2. ДЕКАЛИ, СЛЕДЫ ШИН, НАКАТ, РАЗМЕТКА (Плавная альфа без Alpha Scissor!)
	if s in ["ksperpixelalpha", "ksskidmarks", "ksclouds", "kswindscreen", "ksperpixel_nospec_alpha", "smglass", "ksperpixelreflection_alpha"] or blend_mode > 0:
		return ShaderFamily.DECAL_ALPHA
	for kw in DECAL_KEYWORDS:
		if kw in m:
			return ShaderFamily.DECAL_ALPHA

	# 3. РАСТИТЕЛЬНОСТЬ И ДЕРЕВЬЯ (Вертикальные нормали + Backlight)
	if s.begins_with("kstree") or s.begins_with("ksgrass") or s.begins_with("kstree_pps"):
		return ShaderFamily.TREE_FOLIAGE
	for kw in TREE_KEYWORDS:
		if kw in m:
			return ShaderFamily.TREE_FOLIAGE
	for slot_val in tex_slots.values():
		var t_low = str(slot_val).to_lower()
		for kw in TREE_KEYWORDS:
			if kw in t_low:
				return ShaderFamily.TREE_FOLIAGE

	# 4. ВЫРЕЗНАЯ ПРОЗРАЧНОСТЬ: ЗАБОРЫ, СЕТКИ, ПРОВОЛОКА (Alpha Cutout + A2C)
	if "at" in s or alpha_tested:
		return ShaderFamily.ALPHA_CUTOUT

	# 5. МУЛЬТИМАП (База + 1 детальная текстура)
	if s.begins_with("ksperpixelmultimap") or s.begins_with("stperpixelmultimap") or tex_slots.has("txDetail"):
		return ShaderFamily.MULTIMAP

	# 6. СТАНДАРТНЫЙ НЕПРОЗРАЧНЫЙ PBR (Все остальные ~30 шейдеров AC)
	return ShaderFamily.OPAQUE_PBR

