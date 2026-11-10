shader_type spatial;
render_mode blend_mix, depth_draw_opaque, cull_back, diffuse_burley, specular_schlick_ggx;

uniform sampler2D txDiffuse : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D txNormal : hint_normal, filter_linear_mipmap_anisotropic, repeat_enable;

uniform float ksAmbient = 0.3;
uniform float ksDiffuse = 0.3;
uniform float ksSpecular = 0.1;

void fragment() {
	// Строго родные UV меша (UV * 1.0)
	vec3 base_rgb = texture(txDiffuse, UV).rgb;
	float light_scale = clamp((ksAmbient + ksDiffuse) * 1.6, 0.6, 1.2);
	
	ALBEDO = base_rgb * light_scale;
	ALPHA = 1.0;
	ROUGHNESS = clamp(1.0 - ksSpecular, 0.85, 1.0);
	SPECULAR = 0.05;
	METALLIC = 0.0;
	NORMAL_MAP = texture(txNormal, UV).rgb;
}
