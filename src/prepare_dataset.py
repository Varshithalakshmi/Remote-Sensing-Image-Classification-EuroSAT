from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "data" / "raw" / "EuroSAT_RGB"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

image_paths = []
labels = []

for class_folder in sorted(DATASET_DIR.iterdir()):
    if class_folder.is_dir():
        for image_path in sorted(class_folder.glob("*.jpg")):
            image_paths.append(
                str(image_path.relative_to(PROJECT_DIR))
            )
            labels.append(class_folder.name)

df = pd.DataFrame({
    "image_path": image_paths,
    "label": labels
})

train_df, temporary_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["label"]
)

validation_df, test_df = train_test_split(
    temporary_df,
    test_size=1/3,
    random_state=42,
    stratify=temporary_df["label"]
)

train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
validation_df.to_csv(OUTPUT_DIR / "validation.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

print("\nDATASET SPLITTING COMPLETED")
print("=" * 40)
print(f"Training images:   {len(train_df)}")
print(f"Validation images: {len(validation_df)}")
print(f"Testing images:    {len(test_df)}")
print(f"Total images:      {len(df)}")
print("=" * 40)

print("\nTraining class distribution:")
print(train_df["label"].value_counts().sort_index())

print("\nCSV files saved in:", OUTPUT_DIR)
