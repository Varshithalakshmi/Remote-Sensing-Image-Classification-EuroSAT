from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "data" / "raw" / "EuroSAT_RGB"

classes = sorted(
    folder.name for folder in DATASET_DIR.iterdir()
    if folder.is_dir()
)

plt.figure(figsize=(15, 10))

for index, class_name in enumerate(classes):
    image_path = next((DATASET_DIR / class_name).glob("*.jpg"))
    image = Image.open(image_path)

    plt.subplot(2, 5, index + 1)
    plt.imshow(image)
    plt.title(class_name)
    plt.axis("off")

plt.tight_layout()

output_path = PROJECT_DIR / "results" / "dataset_samples.png"
plt.savefig(output_path, dpi=150)
plt.show()

print(f"Image saved at: {output_path}")
