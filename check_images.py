from PIL import Image
import os

deleted = 0

for root, dirs, files in os.walk("data"):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(root, file)
            try:
                with Image.open(path) as img:
                    img.verify()
            except Exception:
                print("DELETING CORRUPTED:", path)
                try:
                    os.remove(path)
                    deleted += 1
                except Exception as e:
                    print("Could not delete:", path, e)

print(f"\n✅ Done. Total corrupted images deleted: {deleted}")
