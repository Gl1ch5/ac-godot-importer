shader_type spatial;
render_mode blend_mix, depth_draw_opaque, cull_back, diffuse_burley, specular_schlick_ggx;

uniform sampler2D txDiffuse : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D txNormal : hint_normal, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D txMaps : filter_linear_mipmap, repeat_enable;
uniform sampler2D txDetail : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform bool has_detail = false;
uniform float detailUVMultiplier = 1.0;

uniform float ksAmbient = 0.3;
uniform float ksDiffuse = 0.3;
uniform float ksSpecular = 0.1;

void fragment() {
	vec4 diff = texture(txDiffuse, UV);
	vec3 base_rgb = diff.rgb;
	
	if (has_detail) {
		float uv_mult = detailUVMultiplier > 0.0 ? detailUVMultiplier : 1.0;
		vec3 det = texture(txDetail, UV * uv_mult).rgb;
		base_rgb = base_rgb * det;
	}
	
	float light_scale = clamp(ksAmbient + ksDiffuse * 1.2, 0.35, 1.0);
	ALBEDO = base_rgb * light_scale;
	
	vec4 maps = texture(txMaps, UV);
	SPECULAR = maps.r * ksSpecular;
	ROUGHNESS = clamp(1.0 - maps.g, 0.05, 0.98);
	METALLIC = maps.b;
	
	NORMAL_MAP = texture(txNormal, UV).rgb;
}
