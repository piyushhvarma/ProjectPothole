from PIL import Image
import os

folder = "route_photos"
images = sorted(os.listdir(folder))

print(f"{'Filename':<40} | {'Size (KB)':<10} | {'Mode':<5} | {'Size (px)':<10}")
print("-" * 75)

for img in images:
    path = os.path.join(folder, img)
    s = os.path.getsize(path) / 1024
    try:
        with Image.open(path) as i:
            print(f"{img:<40} | {s:<10.2f} | {i.mode:<5} | {i.size[0]}x{i.size[1]}")
    except:
        print(f"{img:<40} | {s:<10.2f} | ERROR")
