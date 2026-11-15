from pathlib import Path
from PIL import Image

# Dataset location
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "data" / "raw" / "EuroSAT_RGB"

def inspect_dataset():
    print("\nEUROSAT DATASET INSPECTION")
    print("=" * 45)

    if not DATASET_DIR.exists():
        print("Dataset folder not found!")
        return

    total_images = 0

    for class_folder in sorted(DATASET_DIR.iterdir()):
        if class_folder.is_dir():
            images = list(class_folder.glob("*.jpg"))
            count = len(images)
            total_images += count

            if images:
                with Image.open(images[0]) as img:
                    dimensions = img.size

                print(
                    f"{class_folder.name:25} "
                    f"{count:5} images | Size: {dimensions}"
                )

    print("=" * 45)
    print(f"Total images: {total_images}")

if __name__ == "__main__":
    inspect_dataset()
