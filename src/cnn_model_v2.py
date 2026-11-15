
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_cnn_model_v2(num_classes=10):

    inputs = layers.Input(shape=(64, 64, 3))

    # Data augmentation for all satellite image categories
    x = layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = layers.RandomRotation(0.10)(x)
    x = layers.RandomZoom(0.10)(x)

    # Block 1
    x = layers.Conv2D(
        32, (3, 3), padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(
        32, (3, 3), padding="same", activation="relu"
    )(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.15)(x)

    # Block 2
    x = layers.Conv2D(
        64, (3, 3), padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(
        64, (3, 3), padding="same", activation="relu"
    )(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.20)(x)

    # Block 3
    x = layers.Conv2D(
        128, (3, 3), padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(
        128, (3, 3), padding="same", activation="relu"
    )(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 4
    x = layers.Conv2D(
        256, (3, 3), padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(0.001)
    )(x)

    x = layers.Dropout(0.40)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="EuroSAT_Custom_CNN_V2"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0005
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    model = build_cnn_model_v2()
    model.summary()
    print("Custom CNN V2 created successfully!")
