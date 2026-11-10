shader_type spatial;
render_mode blend_mix, depth_draw_opaque, cull_disabled, alpha_to_coverage, diffuse_burley, specular_schlick_ggx;

uniform sampler2D txDiffuse : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform float ksAlphaRef = 0.3;
uniform float ksAmbient = 0.4;
uniform float ksDiffuse = 0.4;
uniform float ksSpecular = 0.1;

void fragment() {
	vec4 tex = texture(txDiffuse, UV);
	vec3 up_normal = normalize((VIEW_MATRIX * vec4(0.0, 1.0, 0.0, 0.0)).xyz);
	NORMAL = FRONT_FACING ? up_normal : -up_normal;
	
	ALBEDO = tex.rgb * clamp(ksAmbient + ksDiffuse * 1.2, 0.65, 1.0);
	ALPHA = tex.a;
	ALPHA_SCISSOR_THRESHOLD = clamp(ksAlphaRef * 0.6, 0.12, 0.45);
	ROUGHNESS = clamp(1.0 - ksSpecular, 0.6, 0.95);
	SPECULAR = ksSpecular;
	BACKLIGHT = vec3(0.45, 0.5, 0.4);
}
