cat > prepare_negative_data.py <<'PY'
from pathlib import Path
from tensorflow.keras.datasets import cifar10
from PIL import Image

BASE = Path("data/non_satellite")

for folder in ["animals", "ordinary_photos"]:
    (BASE / folder).mkdir(parents=True, exist_ok=True)

print("Downloading CIFAR-10 dataset...")

(x_train, y_train), _ = cifar10.load_data()

animal_labels = [2, 3, 4, 5, 6, 7]
ordinary_labels = [0, 1, 8, 9]

counts = {"animals": 0, "ordinary_photos": 0}

for image, label in zip(x_train, y_train.flatten()):
    if label in animal_labels:
        category = "animals"
    elif label in ordinary_labels:
        category = "ordinary_photos"
    else:
        continue

    if counts[category] >= 1000:
        continue

    output = BASE / category / f"image_{counts[category]}.jpg"
    Image.fromarray(image).save(output, quality=95)
    counts[category] += 1

    if all(value >= 1000 for value in counts.values()):
        break

print("Download completed!")
print(counts)
PY give me thsi