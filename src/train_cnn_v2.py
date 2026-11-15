
from pathlib import Path
import tensorflow as tf

from src.data_loader import create_dataset
from src.cnn_model_v2 import build_cnn_model_v2

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Training settings
EPOCHS = 20

print("Loading EuroSAT datasets...")

train_dataset = create_dataset("train.csv", shuffle=True)
validation_dataset = create_dataset("validation.csv", shuffle=False)

print("Building Custom CNN V2...")

model = build_cnn_model_v2(num_classes=10)

model.summary()

# Save V2 separately
best_model_path = MODEL_DIR / "best_cnn_v2_model.keras"
final_model_path = MODEL_DIR / "final_cnn_v2_model.keras"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=best_model_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=0.000001,
        verbose=1
    )
]

print("\nStarting Custom CNN V2 training...")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Save final model separately
model.save(final_model_path)

print("\nCustom CNN V2 training completed!")
print("Best model:", best_model_path)
print("Final model:", final_model_path)

best_accuracy = max(history.history["val_accuracy"])
print(f"Best validation accuracy: {best_accuracy * 100:.2f}%")
