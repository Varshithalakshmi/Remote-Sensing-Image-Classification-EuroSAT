import tensorflow as tf
from tensorflow.keras import layers, models


def residual_block(x, filters, stride=1):
    shortcut = x

    x = layers.Conv2D(
        filters, 3, strides=stride, padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(
        filters, 3, strides=1, padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters, 1, strides=stride, padding="same", use_bias=False
        )(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    return layers.ReLU()(x)


def build_resnet18(num_classes=10):
    inputs = layers.Input(shape=(64, 64, 3))

    # Small-image stem adapted for 64x64 EuroSAT images
    x = layers.Conv2D(
        64, 3, strides=1, padding="same", use_bias=False
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Four ResNet stages, with two residual blocks each
    for filters, stride in [(64, 1), (128, 2), (256, 2), (512, 2)]:
        x = residual_block(x, filters, stride)
        x = residual_block(x, filters, 1)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="ResNet18")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
