"""
CTC-based lip reading model (TensorFlow/Keras).

Architecture: 3D-CNN front-end → Bidirectional GRU → per-frame character prediction
Uses CTC loss (via keras.backend.ctc_batch_cost) for alignment-free sequence prediction.
"""

import tensorflow as tf
from keras import layers, Model, backend as K

from src.utils import MAX_FRAMES, BLANK_IDX, NUM_CHARS


class LipReadingCTC(Model):
    """
    3D-CNN + BiGRU lip reading model with CTC output.

    Input:  (batch, T=75, H=80, W=120, 1) — grayscale video frames
    Output: (batch, T=75, num_chars) — character logits per time step
    """

    def __init__(self, num_chars: int = NUM_CHARS, **kwargs):
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
        # Per-frame character output
        # -----------------------------------------------------------
        self.char_dense = layers.Dense(128, activation="relu")
        self.char_dropout = layers.Dropout(0.3)
        # Softmax output — ctc_batch_cost expects probabilities
        self.char_output = layers.Dense(num_chars, activation="softmax", name="char_probs")

    def call(self, inputs, training=False):
        # 3D CNN
        x = self.pool1(self.relu1(self.bn1(self.conv1(inputs), training=training)))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x), training=training)))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x), training=training)))

        x = self.time_flatten(x)

        # BiGRU
        x = self.bigru1(x, training=training)
        x = self.bigru2(x, training=training)

        # Per-frame character probabilities
        x = self.char_dense(x)
        x = self.char_dropout(x, training=training)
        probs = self.char_output(x)  # (B, T, num_chars)

        return probs

    def _compute_ctc_loss(self, y_pred, labels, label_length):
        """Compute CTC loss using keras backend."""
        batch_size = tf.shape(y_pred)[0]
        # ctc_batch_cost expects input_length as (batch, 1) and label_length as (batch, 1)
        input_length = tf.fill([batch_size, 1], MAX_FRAMES)
        label_length = tf.cast(tf.reshape(label_length, [-1, 1]), tf.int32)

        # ctc_batch_cost expects y_pred as (batch, time, classes) — softmax probs
        # and y_true as (batch, max_label_length) — integer labels
        loss = K.ctc_batch_cost(
            y_true=tf.cast(labels, tf.float32),
            y_pred=y_pred,
            input_length=tf.cast(input_length, tf.int32),
            label_length=label_length,
        )
        return tf.reduce_mean(loss)

    def train_step(self, data):
        """Custom training step with CTC loss."""
        x, y = data
        labels = y["labels"]
        label_length = y["label_length"]

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self._compute_ctc_loss(y_pred, labels, label_length)

        grads = tape.gradient(loss, self.trainable_variables)
        # Clip gradients — critical for CTC stability
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {"loss": loss}

    def test_step(self, data):
        """Custom test step with CTC loss."""
        x, y = data
        labels = y["labels"]
        label_length = y["label_length"]

        y_pred = self(x, training=False)
        loss = self._compute_ctc_loss(y_pred, labels, label_length)

        return {"loss": loss}

    def decode_greedy(self, y_pred):
        """
        Greedy CTC decoding.

        Args:
            y_pred: (batch, T, num_chars) — softmax probabilities

        Returns:
            List of decoded index sequences (one per batch item)
        """
        # Convert to log-probs, transpose to time-major
        log_probs = tf.math.log(tf.clip_by_value(y_pred, 1e-7, 1.0))
        logits_t = tf.transpose(log_probs, [1, 0, 2])  # (T, B, C)
        input_length = tf.fill([tf.shape(y_pred)[0]], MAX_FRAMES)

        # Blank is the LAST class for keras CTC
        decoded, _ = tf.nn.ctc_greedy_decoder(
            logits_t, input_length, blank_index=BLANK_IDX
        )
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

    model = LipReadingCTC(num_chars=NUM_CHARS)
    dummy = np.random.randn(2, 75, 80, 120, 1).astype(np.float32)
    probs = model(dummy)

    print(f"Output shape: {probs.shape}")  # (2, 75, 28)
    print(f"Sum per frame (should be ~1.0): {probs[0, 0].numpy().sum():.4f}")
    print(f"Trainable parameters: {count_parameters(model):,}")
