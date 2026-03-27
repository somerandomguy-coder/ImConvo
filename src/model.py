"""
CTC-based lip reading model (TensorFlow/Keras).

Architecture: 3D-CNN front-end → Bidirectional GRU → per-frame character prediction
Uses CTC loss for alignment-free sequence prediction.
"""

import tensorflow as tf
from keras import layers, Model

from src.utils import MAX_FRAMES, BLANK_IDX


class LipReadingCTC(Model):
    """
    3D-CNN + BiGRU lip reading model with CTC output.

    Input:  (batch, T=75, H=80, W=120, 1) — grayscale video frames
    Output: (batch, T=75, num_chars) — character logits per time step
    """

    def __init__(self, num_chars: int = 28, **kwargs):
        super().__init__(name="LipReadingCTC", **kwargs)
        self.num_chars = num_chars

        # -----------------------------------------------------------
        # 3D CNN front-end
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

        self.time_flatten = layers.TimeDistributed(layers.Flatten())

        # -----------------------------------------------------------
        # Bidirectional GRU (both return sequences for CTC)
        # -----------------------------------------------------------
        self.bigru1 = layers.Bidirectional(
            layers.GRU(256, return_sequences=True, dropout=0.3)
        )
        self.bigru2 = layers.Bidirectional(
            layers.GRU(256, return_sequences=True, dropout=0.3)
        )

        # -----------------------------------------------------------
        # Per-frame character output (logits, no softmax — CTC needs raw logits)
        # -----------------------------------------------------------
        self.char_dense = layers.Dense(128, activation="relu")
        self.char_dropout = layers.Dropout(0.3)
        self.char_output = layers.Dense(num_chars, name="char_logits")

    def call(self, inputs, training=False):
        """
        Forward pass.

        Args:
            inputs: (batch, T, H, W, 1)

        Returns:
            logits: (batch, T, num_chars) — raw logits per frame
        """
        # 3D CNN
        x = self.pool1(self.relu1(self.bn1(self.conv1(inputs), training=training)))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x), training=training)))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x), training=training)))

        # Flatten spatial → (B, T, features)
        x = self.time_flatten(x)

        # BiGRU (both return sequences for per-frame output)
        x = self.bigru1(x, training=training)
        x = self.bigru2(x, training=training)

        # Per-frame character logits
        x = self.char_dense(x)
        x = self.char_dropout(x, training=training)
        logits = self.char_output(x)  # (B, T, num_chars)

        return logits

    def train_step(self, data):
        """Custom training step with CTC loss."""
        x, y = data
        labels = y["labels"]            # (B, max_label_len)
        label_length = y["label_length"]  # (B,)

        with tf.GradientTape() as tape:
            logits = self(x, training=True)  # (B, T, num_chars)

            batch_size = tf.shape(logits)[0]
            input_length = tf.fill([batch_size], MAX_FRAMES)

            loss = tf.nn.ctc_loss(
                labels=labels,
                logits=logits,
                label_length=label_length,
                logit_length=input_length,
                logits_time_major=False,
                blank_index=BLANK_IDX,
            )
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {"loss": loss}

    def test_step(self, data):
        """Custom test step with CTC loss."""
        x, y = data
        labels = y["labels"]
        label_length = y["label_length"]

        logits = self(x, training=False)

        batch_size = tf.shape(logits)[0]
        input_length = tf.fill([batch_size], MAX_FRAMES)

        loss = tf.nn.ctc_loss(
            labels=labels,
            logits=logits,
            label_length=label_length,
            logit_length=input_length,
            logits_time_major=False,
            blank_index=BLANK_IDX,
        )

        return {"loss": tf.reduce_mean(loss)}

    def decode_greedy(self, logits):
        """
        Greedy CTC decoding.

        Args:
            logits: (batch, T, num_chars)

        Returns:
            List of decoded index sequences (one per batch item)
        """
        # Transpose to time-major for CTC decoder
        logits_t = tf.transpose(logits, [1, 0, 2])  # (T, B, C)
        input_length = tf.fill([tf.shape(logits)[0]], MAX_FRAMES)

        decoded, _ = tf.nn.ctc_greedy_decoder(
            logits_t, input_length, blank_index=BLANK_IDX
        )
        # decoded[0] is a SparseTensor
        sparse = decoded[0]
        dense = tf.sparse.to_dense(sparse, default_value=-1)
        return dense.numpy()

    def get_config(self):
        config = super().get_config()
        config.update({"num_chars": self.num_chars})
        return config


def count_parameters(model: Model) -> int:
    """Count trainable parameters."""
    return int(sum(tf.reduce_prod(w.shape) for w in model.trainable_weights))


if __name__ == "__main__":
    import numpy as np

    model = LipReadingCTC(num_chars=28)
    dummy = np.random.randn(2, 75, 80, 120, 1).astype(np.float32)
    logits = model(dummy)

    print(f"Logits shape: {logits.shape}")  # (2, 75, 28)
    print(f"Trainable parameters: {count_parameters(model):,}")

    # Test greedy decoding
    decoded = model.decode_greedy(logits)
    print(f"Decoded shape: {decoded.shape}")
