"""
CNN model for lip reading on the GRID corpus (TensorFlow/Keras version).

Architecture: 3D-CNN front-end → Bidirectional GRU → word classification
at each of the 6 positions in the GRID sentence structure.
"""

import tensorflow as tf
from keras import layers, Model


class LipReadingCNN(Model):
    """
    3D-CNN + BiGRU lip reading model.

    Input:  (batch, T=75, H=80, W=120, 1) — grayscale video frames
    Output: dict of 6 tensors, each (batch, num_classes) — one per sentence slot
    """

    SLOT_NAMES = ["command", "color", "preposition", "letter", "digit", "adverb"]

    def __init__(self, num_classes: int, num_slots: int = 6, **kwargs):
        super().__init__(name="LipReadingCNN", **kwargs)
        self.num_classes = num_classes
        self.num_slots = num_slots

        # -----------------------------------------------------------
        # 3D CNN front-end: extract spatiotemporal features
        # -----------------------------------------------------------
        self.conv1 = layers.Conv3D(32, kernel_size=(3, 5, 5), strides=(1, 2, 2), padding="same")
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.pool1 = layers.MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2))

        self.conv2 = layers.Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same")
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()
        self.pool2 = layers.MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2))

        self.conv3 = layers.Conv3D(96, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same")
        self.bn3 = layers.BatchNormalization()
        self.relu3 = layers.ReLU()
        self.pool3 = layers.MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2))

        # Flatten spatial dims per time step
        self.time_flatten = layers.TimeDistributed(layers.Flatten())

        # -----------------------------------------------------------
        # Temporal modelling: Bidirectional GRU
        # -----------------------------------------------------------
        self.bigru1 = layers.Bidirectional(
            layers.GRU(256, return_sequences=True, dropout=0.3)
        )
        self.bigru2 = layers.Bidirectional(
            layers.GRU(256, return_sequences=False, dropout=0.3)
        )

        # -----------------------------------------------------------
        # Shared dense + per-slot classification heads
        # -----------------------------------------------------------
        self.shared_dense = layers.Dense(128, activation="relu")
        self.shared_dropout = layers.Dropout(0.3)

        self.heads = {
            f"slot_{i}": layers.Dense(num_classes, activation="softmax", name=f"slot_{i}")
            for i in range(num_slots)
        }

    def call(self, inputs, training=False):
        # 3D CNN front-end
        x = self.pool1(self.relu1(self.bn1(self.conv1(inputs), training=training)))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x), training=training)))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x), training=training)))

        # Flatten spatial → (B, T, C*H'*W')
        x = self.time_flatten(x)

        # BiGRU temporal modelling
        x = self.bigru1(x, training=training)
        x = self.bigru2(x, training=training)

        # Shared classifier
        x = self.shared_dense(x)
        x = self.shared_dropout(x, training=training)

        # Per-slot predictions
        return {name: head(x) for name, head in self.heads.items()}

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "num_slots": self.num_slots,
        })
        return config

    def build_graph(self, max_frames=75, frame_height=80, frame_width=120):
        """Build the model graph for summary/visualization."""
        x = layers.Input(shape=(max_frames, frame_height, frame_width, 1), name="video_input")
        return Model(inputs=x, outputs=self.call(x))


def count_parameters(model: Model) -> int:
    """Count trainable parameters."""
    return int(sum(tf.reduce_prod(w.shape) for w in model.trainable_weights))


if __name__ == "__main__":
    import numpy as np

    model = LipReadingCNN(num_classes=51)
    dummy = np.random.randn(2, 75, 80, 120, 1).astype(np.float32)
    out = model(dummy)

    print(f"Trainable parameters: {count_parameters(model):,}")
    for name, tensor in out.items():
        print(f"{name}: {tensor.shape}")
    model.build_graph().summary()
