shader_type spatial;
render_mode blend_mix, depth_draw_opaque, cull_disabled, alpha_to_coverage, diffuse_burley, specular_disabled;

uniform sampler2D txDiffuse : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform float ksAlphaRef = 0.3;
uniform float ksAmbient = 0.3;
uniform float ksDiffuse = 0.2;

void vertex() {
	NORMAL = normalize(MODEL_NORMAL_MATRIX * vec3(0.0, 1.0, 0.0));
}

void fragment() {
	vec4 tex = texture(txDiffuse, UV);
	NORMAL = (VIEW_MATRIX * vec4(0.0, 1.0, 0.0, 0.0)).xyz;
	if (!FRONT_FACING) {
		NORMAL = -NORMAL;
	}
	float tone = clamp((ksAmbient + ksDiffuse) * 1.4, 0.45, 1.0);
	
	ALBEDO = tex.rgb * tone;
	ALPHA = tex.a;
	ALPHA_SCISSOR_THRESHOLD = clamp(ksAlphaRef, 0.2, 0.5);
	ROUGHNESS = 0.95;
	SPECULAR = 0.0;
	BACKLIGHT = vec3(0.45, 0.5, 0.4);
}
