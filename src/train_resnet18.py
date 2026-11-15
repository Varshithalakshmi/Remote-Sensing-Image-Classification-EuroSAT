import tensorflow as tf
from src.data_loader import create_dataset
from src.resnet18_model import build_resnet18

EPOCHS = 15

print("Loading EuroSAT datasets...")

train_dataset = create_dataset("train.csv", shuffle=True)
validation_dataset = create_dataset("validation.csv", shuffle=False)

print("Building ResNet18 model...")

model = build_resnet18(num_classes=10)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "models/best_resnet18_model.keras",
        monitor="val_accuracy",
        save_best_only=True
    )
]

print("Starting ResNet18 training...")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)

model.save("models/final_resnet18_model.keras")

print("ResNet18 training completed!")
