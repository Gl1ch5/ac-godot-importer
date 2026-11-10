shader_type spatial;
render_mode blend_mix, depth_draw_opaque, cull_disabled, alpha_to_coverage, diffuse_burley, specular_schlick_ggx;

uniform sampler2D txDiffuse : source_color, filter_linear_mipmap_anisotropic, repeat_enable;

uniform float ksAlphaRef = 0.3;
uniform float ksAmbient = 0.4;
uniform float ksDiffuse = 0.4;
uniform float ksSpecular = 0.1;

void fragment() {
	vec4 tex = texture(txDiffuse, UV);
	ALBEDO = tex.rgb * clamp(ksAmbient + ksDiffuse * 1.2, 0.6, 1.0);
	ALPHA = tex.a;
	ALPHA_SCISSOR_THRESHOLD = clamp(ksAlphaRef, 0.25, 0.6);
	ROUGHNESS = 0.85;
	SPECULAR = 0.1;
}
