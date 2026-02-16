from PIL import Image
import os

deleted = 0

for root, _, files in os.walk("data"):
    for file in files:
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(root, file)

        try:
            with Image.open(path) as img:
                img.verify()

            with Image.open(path) as img:
                channels = len(img.getbands())
                if channels not in (1, 3, 4):
                    print("❌ Invalid channels:", channels, "→ deleting", path)
                    os.remove(path)
                    deleted += 1

        except Exception:
            print("❌ Corrupted image → deleting", path)
            os.remove(path)
            deleted += 1

print(f"\n✅ Cleanup done. Deleted {deleted} bad images.")
