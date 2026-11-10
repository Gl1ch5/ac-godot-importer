shader_type spatial;
render_mode blend_mix, depth_draw_never, cull_back, diffuse_lambert, specular_disabled;

uniform sampler2D txDiffuse : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform float ksAmbient = 0.15;
uniform float ksDiffuse = 0.1;
uniform float ksSpecular = 0.0;
uniform float alpha = 1.0;
uniform float z_offset = 0.002;

void vertex() {
	VERTEX += NORMAL * z_offset;
}

void fragment() {
	vec4 tex = texture(txDiffuse, UV);
	float darkness = clamp(ksDiffuse + ksAmbient, 0.08, 1.0);
	
	ALBEDO = tex.rgb * darkness;
	ALPHA = tex.a * alpha * COLOR.a;
	ROUGHNESS = clamp(1.0 - ksSpecular, 0.1, 1.0);
	SPECULAR = ksSpecular;
	METALLIC = 0.0;
}
