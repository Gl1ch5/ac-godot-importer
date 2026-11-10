class_name ACMaterialFactory
extends RefCounted

const ACTextureLoader = preload("res://materials/ac_texture_loader.gd")
const ACShaderRouter = preload("res://materials/ac_shader_router.gd")

var tex_loader: RefCounted
var shaders = {
	ACShaderRouter.ShaderFamily.MULTILAYER: preload("res://shaders/ac_multilayer.gdshader"),
	ACShaderRouter.ShaderFamily.MULTIMAP: preload("res://shaders/ac_multimap.gdshader"),
	ACShaderRouter.ShaderFamily.TREE_FOLIAGE: preload("res://shaders/ac_tree_foliage.gdshader"),
	ACShaderRouter.ShaderFamily.ALPHA_CUTOUT: preload("res://shaders/ac_alpha_cutout.gdshader"),
	ACShaderRouter.ShaderFamily.DECAL_ALPHA: preload("res://shaders/ac_decal_alpha.gdshader"),
	ACShaderRouter.ShaderFamily.OPAQUE_PBR: preload("res://shaders/ac_opaque_pbr.gdshader"),
}

func _init(p_tex_loader) -> void:
	tex_loader = p_tex_loader

func create_material(info: Dictionary) -> Material:
	var shader_name: String = info.get("shader", "")
	var blend_mode: int = info.get("blend_mode", 0)
	var alpha_tested: bool = info.get("alpha_tested", false)
	var mat_name: String = info.get("name", "")
	var props: Dictionary = info.get("properties", {})
	var texs: Dictionary = info.get("textures", {})

	var family = ACShaderRouter.classify(shader_name, blend_mode, alpha_tested, mat_name, texs)
	var sm = ShaderMaterial.new()
	sm.shader = shaders[family]

	match family:
		ACShaderRouter.ShaderFamily.MULTILAYER:
			# Очищаем альфу у diffuse, чтобы specular mask не давала пятен
			sm.set_shader_parameter("txDiffuse", _tex_or_white(texs.get("txDiffuse", ""), true))
			sm.set_shader_parameter("txMask", _tex_or_white(texs.get("txMask", ""), false))
			sm.set_shader_parameter("txDetailR", _tex_or_white(texs.get("txDetailR", ""), false))
			sm.set_shader_parameter("txDetailG", _tex_or_white(texs.get("txDetailG", ""), false))
			sm.set_shader_parameter("txDetailB", _tex_or_white(texs.get("txDetailB", ""), false))
			sm.set_shader_parameter("txDetailA", _tex_or_white(texs.get("txDetailA", ""), false))
			sm.set_shader_parameter("multR", props.get("multR", 0.6))
			sm.set_shader_parameter("multG", props.get("multG", 0.3))
			sm.set_shader_parameter("multB", props.get("multB", 0.8))
			sm.set_shader_parameter("multA", props.get("multA", 0.0))
			sm.set_shader_parameter("magicMult", props.get("magicMult", 1.0))
			sm.set_shader_parameter("ksAmbient", props.get("ksAmbient", 0.25))
			sm.set_shader_parameter("ksDiffuse", props.get("ksDiffuse", 0.25))
			sm.set_shader_parameter("ksSpecular", props.get("ksSpecular", 0.1))
			sm.set_shader_parameter("is_objsp", shader_name.contains("objsp"))

		ACShaderRouter.ShaderFamily.MULTIMAP:
			sm.set_shader_parameter("txDiffuse", _tex_or_white(texs.get("txDiffuse", ""), false))
			sm.set_shader_parameter("txNormal", _tex_or_normal(texs.get("txNormal", "")))
			sm.set_shader_parameter("txMaps", _tex_or_white(texs.get("txMaps", ""), false))
			var det = tex_loader.get_texture(texs.get("txDetail", ""), false)
			sm.set_shader_parameter("has_detail", det != null)
			sm.set_shader_parameter("txDetail", det if det != null else tex_loader.white_1x1)
			sm.set_shader_parameter("detailUVMultiplier", props.get("detailUVMultiplier", 1.0))
			sm.set_shader_parameter("ksAmbient", props.get("ksAmbient", 0.3))
			sm.set_shader_parameter("ksDiffuse", props.get("ksDiffuse", 0.3))
			sm.set_shader_parameter("ksSpecular", props.get("ksSpecular", 0.1))

		ACShaderRouter.ShaderFamily.TREE_FOLIAGE:
			sm.set_shader_parameter("txDiffuse", _tex_or_white(texs.get("txDiffuse", ""), false))
			sm.set_shader_parameter("ksAlphaRef", props.get("ksAlphaRef", 0.3))
			sm.set_shader_parameter("ksAmbient", props.get("ksAmbient", 0.3))
			sm.set_shader_parameter("ksDiffuse", props.get("ksDiffuse", 0.2))

		ACShaderRouter.ShaderFamily.ALPHA_CUTOUT:
			sm.set_shader_parameter("txDiffuse", _tex_or_white(texs.get("txDiffuse", ""), false))
			sm.set_shader_parameter("ksAlphaRef", props.get("ksAlphaRef", 0.3))
			sm.set_shader_parameter("ksAmbient", props.get("ksAmbient", 0.4))
			sm.set_shader_parameter("ksDiffuse", props.get("ksDiffuse", 0.4))
			sm.set_shader_parameter("ksSpecular", props.get("ksSpecular", 0.1))

		ACShaderRouter.ShaderFamily.DECAL_ALPHA:
			sm.render_priority = 1
			sm.set_shader_parameter("txDiffuse", _tex_or_white(texs.get("txDiffuse", ""), false))
			sm.set_shader_parameter("ksAmbient", props.get("ksAmbient", 0.15))
			sm.set_shader_parameter("ksDiffuse", props.get("ksDiffuse", 0.1))
			sm.set_shader_parameter("ksSpecular", props.get("ksSpecular", 0.0))
			sm.set_shader_parameter("alpha", props.get("alpha", 1.0))
			sm.set_shader_parameter("z_offset", 0.002)

		ACShaderRouter.ShaderFamily.OPAQUE_PBR:
			# Очищаем альфу (force_opaque = true) — предотвращает появление дыр и муара
			sm.set_shader_parameter("txDiffuse", _tex_or_white(texs.get("txDiffuse", ""), true))
			sm.set_shader_parameter("txNormal", _tex_or_normal(texs.get("txNormal", "")))
			sm.set_shader_parameter("ksAmbient", props.get("ksAmbient", 0.3))
			sm.set_shader_parameter("ksDiffuse", props.get("ksDiffuse", 0.3))
			sm.set_shader_parameter("ksSpecular", props.get("ksSpecular", 0.1))

	return sm

func _tex_or_white(tname: String, force_opaque: bool) -> Texture2D:
	var t = tex_loader.get_texture(tname, force_opaque)
	return t if t != null else tex_loader.white_1x1

func _tex_or_normal(tname: String) -> Texture2D:
	var t = tex_loader.get_texture(tname, false)
	return t if t != null else tex_loader.flat_normal_1x1
