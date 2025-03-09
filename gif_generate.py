from PIL import Image
import glob

# Get all PNG files in the directory
image_files = sorted(glob.glob("sample_c2i_*.png"))

if not image_files:
    print("No PNG files found matching the pattern 'LlamaGen/sample_c2i_*.png'")
    exit()

# Open all images
images = [Image.open(f) for f in image_files]
print(len(images))
# Save as GIF
images[0].save(
    "output.gif",
    save_all=True,
    append_images=images[1:],
    duration=100,  # Duration for each frame in milliseconds
    loop=0
)