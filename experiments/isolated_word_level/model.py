"""Model for isolated-word GRID classification."""

from __future__ import annotations

import tensorflow as tf
from keras import layers
import sys, os
sys.path.append(os.getcwd())
from src import FRONTEND_MODELS, MODEL_VARIANTS, LipReadingCTC
from src.utils import NUM_CHARS


class LipReadingIsolatedWordClassifier(LipReadingCTC):
    """
    Reuses existing 3D frontend + temporal backbone, then pools over time
    and predicts a single word class for each clip.
    """

    def __init__(
        self,
        num_word_classes: int,
        model_variant: str = "bigru",
        frontend_model: str = "flatten",
        feature_time_masking: bool = False,
        backbone_dropout: float = 0.3,
        head_dropout: float = 0.3,
        frontend_projection_dim: int = 256,
        classifier_hidden_dim: int = 256,
        **kwargs,
    ):
        super().__init__(
            num_chars=NUM_CHARS,
            model_variant=model_variant,
            frontend_model=frontend_model,
            feature_time_masking=feature_time_masking,
            backbone_dropout=backbone_dropout,
            head_dropout=head_dropout,
            frontend_projection_dim=frontend_projection_dim,
            **kwargs,
        )

        self.num_word_classes = int(num_word_classes)
        self.classifier_hidden_dim = int(classifier_hidden_dim)

        self.temporal_pool = layers.GlobalAveragePooling1D(name="temporal_gap")
        self.classifier_dense = layers.Dense(
            self.classifier_hidden_dim,
            activation="relu",
            name="isolated_classifier_dense",
        )
        self.classifier_dropout = layers.Dropout(self.head_dropout, name="isolated_classifier_dropout")
        self.classifier_logits = layers.Dense(
            self.num_word_classes,
            activation=None,
            name="isolated_word_logits",
        )

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.top1_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name="top1_acc")
        self.top5_tracker = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_acc")

    def _apply_feature_time_mask_dynamic(self, x: tf.Tensor) -> tf.Tensor:
        """Dynamic-time version so it works with shorter isolated clips."""
        seq_len = tf.shape(x)[1]
        batch_size = tf.shape(x)[0]

        max_mask_len = tf.maximum(2, tf.minimum(6, seq_len))
        mask_len = tf.random.uniform([batch_size], minval=1, maxval=max_mask_len + 1, dtype=tf.int32)
        max_start = tf.maximum(1, seq_len - mask_len + 1)

        start = tf.cast(
            tf.random.uniform([batch_size], 0.0, 1.0) * tf.cast(max_start, tf.float32),
            tf.int32,
        )

        time_idx = tf.range(seq_len)[tf.newaxis, :]
        mask = tf.logical_and(
            time_idx >= start[:, tf.newaxis],
            time_idx < (start + mask_len)[:, tf.newaxis],
        )
        return tf.where(mask[:, :, tf.newaxis], tf.zeros_like(x), x)

    def call(self, inputs, training: bool = False):
        x = self._apply_visual_frontend(inputs, training=training)
        if self.feature_time_masking and training:
            x = tf.cond(
                tf.random.uniform([], 0.0, 1.0) < 0.5,
                lambda: self._apply_feature_time_mask_dynamic(x),
                lambda: x,
            )

        x = self._apply_temporal_backbone(x, training=training)
        x = self.temporal_pool(x)
        x = self.classifier_dense(x)
        x = self.classifier_dropout(x, training=training)
        return self.classifier_logits(x)

    def get_head_layers(self) -> list[layers.Layer]:
        return [
            self.temporal_pool,
            self.classifier_dense,
            self.classifier_dropout,
            self.classifier_logits,
        ]

    @property
    def metrics(self):
        return [self.loss_tracker, self.top1_tracker, self.top5_tracker]

    def _compute_loss(self, y_true: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
        loss_vec = tf.keras.losses.sparse_categorical_crossentropy(
            y_true=tf.cast(y_true, tf.int32),
            y_pred=logits,
            from_logits=True,
        )
        return tf.reduce_mean(loss_vec)

    def train_step(self, data):
        x, y = data
        labels = y["label"]

        with tf.GradientTape() as tape:
            logits = self(x, training=True)
            loss = self._compute_loss(labels, logits)

        grads = tape.gradient(loss, self.trainable_variables)
        grads_and_vars = [(g, v) for g, v in zip(grads, self.trainable_variables) if g is not None]
        if grads_and_vars:
            clipped_grads, _ = tf.clip_by_global_norm([g for g, _ in grads_and_vars], 5.0)
            self.optimizer.apply_gradients(zip(clipped_grads, [v for _, v in grads_and_vars]))

        probs = tf.nn.softmax(logits, axis=-1)
        self.loss_tracker.update_state(loss)
        self.top1_tracker.update_state(labels, probs)
        self.top5_tracker.update_state(labels, probs)

        return {
            "loss": self.loss_tracker.result(),
            "top1_acc": self.top1_tracker.result(),
            "top5_acc": self.top5_tracker.result(),
        }

    def test_step(self, data):
        x, y = data
        labels = y["label"]

        logits = self(x, training=False)
        loss = self._compute_loss(labels, logits)
        probs = tf.nn.softmax(logits, axis=-1)

        self.loss_tracker.update_state(loss)
        self.top1_tracker.update_state(labels, probs)
        self.top5_tracker.update_state(labels, probs)

        return {
            "loss": self.loss_tracker.result(),
            "top1_acc": self.top1_tracker.result(),
            "top5_acc": self.top5_tracker.result(),
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_word_classes": self.num_word_classes,
                "classifier_hidden_dim": self.classifier_hidden_dim,
            }
        )
        return config



def build_lipreading_isolated_word_classifier(
    model_variant: str,
    frontend_model: str,
    num_word_classes: int,
    feature_time_masking: bool = False,
    backbone_dropout: float = 0.3,
    head_dropout: float = 0.3,
    frontend_projection_dim: int = 256,
    classifier_hidden_dim: int = 256,
) -> LipReadingIsolatedWordClassifier:
    return LipReadingIsolatedWordClassifier(
        num_word_classes=num_word_classes,
        model_variant=model_variant,
        frontend_model=frontend_model,
        feature_time_masking=feature_time_masking,
        backbone_dropout=backbone_dropout,
        head_dropout=head_dropout,
        frontend_projection_dim=frontend_projection_dim,
        classifier_hidden_dim=classifier_hidden_dim,
    )


__all__ = [
    "LipReadingIsolatedWordClassifier",
    "build_lipreading_isolated_word_classifier",
    "FRONTEND_MODELS",
    "MODEL_VARIANTS",
]
