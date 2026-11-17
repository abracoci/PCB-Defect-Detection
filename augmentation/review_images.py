
from pathlib import Path
import cv2
import shutil

source = Path("output_augmented")
output = Path("output_review")

output.mkdir(exist_ok=True)

count = 0

for path in source.rglob("*.png"):

    image = cv2.imread(str(path))

    if image is None:
        print("ERROR:", path)
        continue

    h, w = image.shape[:2]

    scale = min(1.0, 800 / max(h, w))

    is_mask = path.stem.endswith("_valid_mask")

    resized = cv2.resize(
        image,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=(
            cv2.INTER_NEAREST if is_mask
            else cv2.INTER_AREA
        )
    )

    destination = output / path.relative_to(source)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if cv2.imwrite(str(destination), resized):
        count += 1
    else:
        print("ERROR writing:", destination)

shutil.copy2(
    source / "manifest.csv",
    output / "manifest.csv"
)

print(f"Review files generated: {count}")
print("Original dataset remains unchanged.")
