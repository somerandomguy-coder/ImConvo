"""Model for isolated-word GRID classification."""

from __future__ import annotations

import tensorflow as tf
from keras import layers
from src import FRONTEND_MODELS, MODEL_VARIANTS, LipReadingCTC
from src.utils import NUM_CHARS


class LipReadingIsolatedWordClassifier(LipReadingCTC):
    """Reuses existing 3D frontend + temporal backbone, then pools over time

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

        self.pool = layers.GlobalAveragePooling1D(name="time_pooling")
        self.fc_hidden = layers.Dense(
            self.classifier_hidden_dim,
            activation="relu",
            name="classifier_hidden",
        )
        self.fc_dropout = layers.Dropout(self.head_dropout, name="classifier_dropout")
        self.fc_out = layers.Dense(
            self.num_word_classes,
            activation=None,
            name="classifier_logits",
        )

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.acc_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name="acc")
        self.top5_acc_tracker = tf.keras.metrics.SparseTopKCategoricalAccuracy(
            k=5, name="top5_acc"
        )

    def get_head_layers(self) -> list[layers.Layer]:
        return [self.pool, self.fc_hidden, self.fc_dropout, self.fc_out]

    def call(self, inputs, training: bool = False):
        x = self._apply_visual_frontend(inputs, training=training)
        if self.feature_time_masking and training:
            x = tf.cond(
                tf.random.uniform([], 0.0, 1.0) < 0.5,
                lambda: self._apply_feature_time_mask(x),
                lambda: x,
            )

        x = self._apply_temporal_backbone(x, training=training)
        x = self.pool(x)
        x = self.fc_hidden(x)
        x = self.fc_dropout(x, training=training)
        logits = self.fc_out(x)
        return logits

    def _compute_loss(self, labels: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true=tf.cast(labels, tf.int32),
            y_pred=logits,
            from_logits=True,
        )
        return tf.reduce_mean(loss)

    def _update_metrics(
        self,
        labels: tf.Tensor,
        logits: tf.Tensor,
        loss: tf.Tensor,
    ):
        self.loss_tracker.update_state(loss)
        probs = tf.nn.softmax(logits, axis=-1)
        self.acc_tracker.update_state(labels, probs)
        self.top5_acc_tracker.update_state(labels, probs)

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.acc_tracker,
            self.top5_acc_tracker,
        ]

    def train_step(self, data):
        x, y = data
        labels = y["word_labels"]

        with tf.GradientTape() as tape:
            logits = self(x, training=True)
            loss = self._compute_loss(labels, logits)

        grads = tape.gradient(loss, self.trainable_variables)
        grads_and_vars = [
            (g, v) for g, v in zip(grads, self.trainable_variables) if g is not None
        ]
        if grads_and_vars:
            clipped = tf.clip_by_global_norm(
                [gv[0] for gv in grads_and_vars], 5.0
            )[0]
            self.optimizer.apply_gradients(
                zip(clipped, [gv[1] for gv in grads_and_vars])
            )

        self._update_metrics(labels, logits, loss)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data):
        x, y = data
        labels = y["word_labels"]

        logits = self(x, training=False)
        loss = self._compute_loss(labels, logits)
        self._update_metrics(labels, logits, loss)
        return {metric.name: metric.result() for metric in self.metrics}

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
    """Factory for isolated-word GRID model variants."""
    return LipReadingIsolatedWordClassifier(
        model_variant=model_variant,
        frontend_model=frontend_model,
        num_word_classes=num_word_classes,
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
