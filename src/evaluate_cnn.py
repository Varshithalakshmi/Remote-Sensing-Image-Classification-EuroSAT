from pathlib import Path
import tensorflow as tf
import numpy as np
import pandas as pd

from src.data_loader import create_dataset, CLASS_NAMES

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "best_cnn_model.keras"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

# Load the trained CNN model
model = tf.keras.models.load_model(MODEL_PATH)

# Load the test dataset
test_dataset = create_dataset("test.csv")

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_dataset)

# Generate predictions
predictions = model.predict(test_dataset)
predicted_labels = np.argmax(predictions, axis=1)

# Get actual labels
test_df = pd.read_csv(
    PROJECT_DIR / "data" / "processed" / "test.csv"
)

actual_labels = np.array([
    CLASS_NAMES.index(label)
    for label in test_df["label"]
])

# Classification report
from sklearn.metrics import classification_report, confusion_matrix

report = classification_report(
    actual_labels,
    predicted_labels,
    target_names=CLASS_NAMES,
    digits=4
)

print("\nCNN TEST RESULTS")
print("=" * 50)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

print("\nCLASSIFICATION REPORT")
print(report)

# Save classification report
with open(RESULTS_DIR / "cnn_classification_report.txt", "w") as file:
    file.write(report)

# Save confusion matrix
matrix = confusion_matrix(actual_labels, predicted_labels)

matrix_df = pd.DataFrame(
    matrix,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
)

matrix_df.to_csv(RESULTS_DIR / "cnn_confusion_matrix.csv")

print("Results saved successfully in:", RESULTS_DIR)
