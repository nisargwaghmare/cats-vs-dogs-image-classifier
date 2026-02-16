
from PIL import Image
import sys

bad = False
for f in sys.argv[1:]:
    try:
        Image.open(f).verify()
    except Exception:
        print("CORRUPTED:", f)
        bad = True

if not bad:
    print("No corrupted images found")
EOF {} +
