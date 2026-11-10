shader_type spatial;
render_mode blend_mix, depth_draw_opaque, cull_back, diffuse_burley, specular_schlick_ggx;

uniform sampler2D txDiffuse : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D txMask : filter_linear_mipmap, repeat_enable;
uniform sampler2D txDetailR : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D txDetailG : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D txDetailB : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D txDetailA : source_color, filter_linear_mipmap_anisotropic, repeat_enable;

uniform float multR = 0.6;
uniform float multG = 0.3;
uniform float multB = 0.8;
uniform float multA = 0.0;
uniform float magicMult = 1.0;

uniform float ksAmbient = 0.2;
uniform float ksDiffuse = 0.2;
uniform float ksSpecular = 0.1;

varying vec3 world_pos;

void vertex() {
	world_pos = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
}

void fragment() {
	vec3 base_col = texture(txDiffuse, UV).rgb;
	vec4 mask = texture(txMask, UV);
	
	float mask_sum = max(dot(mask, vec4(1.0)), 0.0001);
	vec4 w = mask / mask_sum;
	
	vec2 world_uv = world_pos.xz;
	
	vec3 detR = texture(txDetailR, world_uv * multR).rgb;
	vec3 detG = texture(txDetailG, world_uv * multG).rgb;
	vec3 detB = texture(txDetailB, world_uv * multB).rgb;
	vec3 detA = (multA > 0.0001) ? texture(txDetailA, world_uv * multA).rgb : detR;
	
	vec3 detail_combined = detR * w.r + detG * w.g + detB * w.b + detA * w.a;
	vec3 final_color = base_col * detail_combined * (magicMult * 2.0);
	float light_scale = clamp(ksDiffuse + ksAmbient * 1.5, 0.25, 1.0);
	
	ALBEDO = final_color * light_scale;
	ROUGHNESS = clamp(1.0 - ksSpecular, 0.75, 0.98);
	SPECULAR = ksSpecular;
	METALLIC = 0.0;
}
