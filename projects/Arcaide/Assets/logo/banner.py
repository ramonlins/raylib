from PIL import Image

input_path = "banner.png"
output_path = "banner_resized.png"

img = Image.open(input_path)

target_size = (2048, 1152)

resized_img = img.resize(target_size, Image.LANCZOS)

resized_img.save(output_path, format="PNG", optimize=True)