from pathlib import Path
import pandas as pd
import tensorflow as tf

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
NUM_CLASSES = 10

CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

def load_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    return image, label


def create_dataset(csv_file, shuffle=False):
    df = pd.read_csv(PROCESSED_DIR / csv_file)

    image_paths = [
        str(PROJECT_DIR / path)
        for path in df["image_path"]
    ]

    labels = [
        CLASS_NAMES.index(label)
        for label in df["label"]
    ]

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=1000,
            seed=42
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=2
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(1)

    return dataset


if __name__ == "__main__":
    train_dataset = create_dataset(
        "train.csv", shuffle=True
    )

    for images, labels in train_dataset.take(1):
        print("Image batch shape:", images.shape)
        print("Label batch shape:", labels.shape)
        print("Pixel value range:",
              float(tf.reduce_min(images)),
              "to",
              float(tf.reduce_max(images)))

    print("\nImage loading pipeline working successfully!")
