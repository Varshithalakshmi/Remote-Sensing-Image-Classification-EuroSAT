import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.data_loader import create_dataset, CLASS_NAMES

print("Loading test dataset...")
test_dataset = create_dataset("test.csv", shuffle=False)

print("Loading best ResNet18 model...")
model = tf.keras.models.load_model("models/best_resnet18_model.keras")

print("Evaluating model...")
predictions = model.predict(test_dataset)

y_pred = np.argmax(predictions, axis=1)
y_true = np.concatenate([labels.numpy() for _, labels in test_dataset])

accuracy = accuracy_score(y_true, y_pred)
print(f"\nResNet18 Test Accuracy: {accuracy * 100:.2f}%")

report = classification_report(
    y_true, y_pred, target_names=CLASS_NAMES, digits=4
)

print("\nClassification Report:\n")
print(report)

with open("results/resnet18_classification_report.txt", "w") as f:
    f.write(f"Test Accuracy: {accuracy * 100:.2f}%\n\n")
    f.write(report)

matrix = confusion_matrix(y_true, y_pred)
np.savetxt(
    "results/resnet18_confusion_matrix.csv",
    matrix,
    delimiter=",",
    fmt="%d"
)

print("Evaluation completed!")
