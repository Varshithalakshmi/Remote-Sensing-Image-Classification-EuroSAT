from pathlib import Path
import tensorflow as tf

from data_loader import create_dataset
from cnn_model import build_cnn_model

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

train_dataset = create_dataset("train.csv", shuffle=True)
validation_dataset = create_dataset("validation.csv")

model = build_cnn_model()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_DIR / "best_cnn_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )
]

print("\nStarting CNN training...")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10,
    callbacks=callbacks
)

model.save(MODEL_DIR / "final_cnn_model.keras")

print("\nCNN training completed!")
print("Best model saved in:", MODEL_DIR)
