class_name ACTextureLoader
extends RefCounted

var cache: Dictionary = {}
var white_1x1: ImageTexture
var black_1x1: ImageTexture
var flat_normal_1x1: ImageTexture

func _init() -> void:
	# Тёмно-серая заглушка (Color 0.28), чтобы отсутствующая текстура никогда не слепила белым
	var img_w = Image.create(1, 1, false, Image.FORMAT_RGBA8)
	img_w.fill(Color(0.28, 0.28, 0.28, 1.0))
	white_1x1 = ImageTexture.create_from_image(img_w)

	var img_b = Image.create(1, 1, false, Image.FORMAT_RGBA8)
	img_b.fill(Color(0.0, 0.0, 0.0, 1.0))
	black_1x1 = ImageTexture.create_from_image(img_b)

	var img_n = Image.create(1, 1, false, Image.FORMAT_RGBA8)
	img_n.fill(Color(0.5, 0.5, 1.0, 1.0))
	flat_normal_1x1 = ImageTexture.create_from_image(img_n)

func normalize_key(tex_name: String) -> String:
	return tex_name.get_file().get_basename().to_lower()

func get_texture(tex_name: String, force_opaque: bool = false) -> Texture2D:
	if tex_name.is_empty():
		return null
	var key = normalize_key(tex_name)
	var cache_key = key + ("_rgb" if force_opaque else "")
	if cache.has(cache_key):
		return cache[cache_key]

	var path = "res://ac_textures/" + key + ".png"
	if FileAccess.file_exists(path):
		var img = Image.load_from_file(path)
		if img != null:
			if force_opaque and img.get_format() == Image.FORMAT_RGBA8:
				img.convert(Image.FORMAT_RGB8)
			img.generate_mipmaps()
			var it = ImageTexture.create_from_image(img)
			cache[cache_key] = it
			return it
	print("MISSING TEXTURE: ", tex_name, " (looked for ", path, ")")
	return null
