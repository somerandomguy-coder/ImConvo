"""Word-level GRID classifier using the existing visual/temporal architecture family."""

from __future__ import annotations

import tensorflow as tf
from keras import layers

from src import FRONTEND_MODELS, MODEL_VARIANTS, LipReadingCTC
from src.utils import NUM_CHARS

from experiments.word_level_grid.common import NUM_SLOTS, SLOT_NAMES, SLOT_VOCAB_SIZES


class LipReadingWordClassifier(LipReadingCTC):
    """
    Reuses LipReadingCTC frontend/backbone, replaces CTC head with 6 slot heads.

    Input:  (batch, 75, 80, 120, 1)
    Output: list of 6 tensors, each (batch, vocab_size_for_slot)
    """

    def __init__(
        self,
        model_variant: str = "bigru",
        frontend_model: str = "flatten",
        slot_vocab_sizes: tuple[int, ...] = SLOT_VOCAB_SIZES,
        feature_time_masking: bool = False,
        backbone_dropout: float = 0.3,
        head_dropout: float = 0.3,
        frontend_projection_dim: int = 256,
        slot_projection_dim: int = 256,
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
        if len(slot_vocab_sizes) != NUM_SLOTS:
            raise ValueError(
                f"Expected {NUM_SLOTS} slot vocab sizes, got {len(slot_vocab_sizes)}"
            )

        self.slot_vocab_sizes = tuple(int(v) for v in slot_vocab_sizes)
        self.slot_projection_dim = int(slot_projection_dim)

        self.slot_projection = layers.Dense(
            self.slot_projection_dim,
            activation="relu",
            name="slot_projection",
        )
        self.slot_dropout = layers.Dropout(self.head_dropout, name="slot_dropout")
        self.slot_queries = self.add_weight(
            name="slot_queries",
            shape=(NUM_SLOTS, self.slot_projection_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.slot_heads = [
            layers.Dense(vocab_size, activation=None, name=f"{slot_name}_logits")
            for slot_name, vocab_size in zip(SLOT_NAMES, self.slot_vocab_sizes)
        ]

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.slot_loss_trackers = [
            tf.keras.metrics.Mean(name=f"{slot}_loss") for slot in SLOT_NAMES
        ]
        self.slot_acc_trackers = [
            tf.keras.metrics.SparseCategoricalAccuracy(name=f"{slot}_acc")
            for slot in SLOT_NAMES
        ]
        self.sentence_acc_tracker = tf.keras.metrics.Mean(name="sentence_acc")
        self.slot_acc_tracker = tf.keras.metrics.Mean(name="slot_acc")

    def get_head_layers(self) -> list[layers.Layer]:
        return [self.slot_projection, self.slot_dropout, *self.slot_heads]

    def call(self, inputs, training: bool = False):
        x = self._apply_visual_frontend(inputs, training=training)
        if self.feature_time_masking and training:
            x = tf.cond(
                tf.random.uniform([], 0.0, 1.0) < 0.5,
                lambda: self._apply_feature_time_mask(x),
                lambda: x,
            )

        x = self._apply_temporal_backbone(x, training=training)
        x = self.slot_projection(x)

        # slot-aware attention pooling over the temporal axis
        attn_scores = tf.einsum("btd,sd->bts", x, self.slot_queries)
        attn_weights = tf.nn.softmax(attn_scores, axis=1)
        slot_context = tf.einsum("btd,bts->bsd", x, attn_weights)
        slot_context = self.slot_dropout(slot_context, training=training)

        logits = [
            head(slot_context[:, slot_idx, :])
            for slot_idx, head in enumerate(self.slot_heads)
        ]
        return logits

    def _compute_loss(
        self,
        labels: tf.Tensor,
        logits_list: list[tf.Tensor],
    ) -> tuple[tf.Tensor, list[tf.Tensor]]:
        per_slot_losses: list[tf.Tensor] = []

        for slot_idx, logits in enumerate(logits_list):
            slot_labels = tf.cast(labels[:, slot_idx], tf.int32)
            slot_loss = tf.keras.losses.sparse_categorical_crossentropy(
                y_true=slot_labels,
                y_pred=logits,
                from_logits=True,
            )
            per_slot_losses.append(tf.reduce_mean(slot_loss))

        total_loss = tf.add_n(per_slot_losses) / float(NUM_SLOTS)
        return total_loss, per_slot_losses

    def _decode_logits(self, logits_list: list[tf.Tensor]) -> tf.Tensor:
        preds = [
            tf.argmax(slot_logits, axis=-1, output_type=tf.int32)
            for slot_logits in logits_list
        ]
        return tf.stack(preds, axis=1)

    def _update_metrics(
        self,
        labels: tf.Tensor,
        logits_list: list[tf.Tensor],
        total_loss: tf.Tensor,
        per_slot_losses: list[tf.Tensor],
    ):
        self.loss_tracker.update_state(total_loss)

        for slot_idx in range(NUM_SLOTS):
            slot_labels = tf.cast(labels[:, slot_idx], tf.int32)
            self.slot_loss_trackers[slot_idx].update_state(per_slot_losses[slot_idx])
            self.slot_acc_trackers[slot_idx].update_state(
                slot_labels,
                tf.nn.softmax(logits_list[slot_idx], axis=-1),
            )

        pred = self._decode_logits(logits_list)
        labels = tf.cast(labels, tf.int32)

        sentence_correct = tf.reduce_all(tf.equal(pred, labels), axis=1)
        sentence_acc = tf.reduce_mean(tf.cast(sentence_correct, tf.float32))
        slot_acc = tf.reduce_mean(tf.cast(tf.equal(pred, labels), tf.float32))

        self.sentence_acc_tracker.update_state(sentence_acc)
        self.slot_acc_tracker.update_state(slot_acc)

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            *self.slot_loss_trackers,
            *self.slot_acc_trackers,
            self.sentence_acc_tracker,
            self.slot_acc_tracker,
        ]

    def train_step(self, data):
        x, y = data
        labels = y["word_labels"]

        with tf.GradientTape() as tape:
            logits_list = self(x, training=True)
            total_loss, per_slot_losses = self._compute_loss(labels, logits_list)

        grads = tape.gradient(total_loss, self.trainable_variables)
        grads_and_vars = [(g, v) for g, v in zip(grads, self.trainable_variables) if g is not None]
        if grads_and_vars:
            clipped = tf.clip_by_global_norm([gv[0] for gv in grads_and_vars], 5.0)[0]
            self.optimizer.apply_gradients(zip(clipped, [gv[1] for gv in grads_and_vars]))

        self._update_metrics(labels, logits_list, total_loss, per_slot_losses)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data):
        x, y = data
        labels = y["word_labels"]

        logits_list = self(x, training=False)
        total_loss, per_slot_losses = self._compute_loss(labels, logits_list)
        self._update_metrics(labels, logits_list, total_loss, per_slot_losses)
        return {metric.name: metric.result() for metric in self.metrics}

    def decode_greedy_slots(self, logits_list: list[tf.Tensor]) -> tf.Tensor:
        return self._decode_logits(logits_list)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "slot_vocab_sizes": list(self.slot_vocab_sizes),
                "slot_projection_dim": self.slot_projection_dim,
            }
        )
        return config


def build_lipreading_word_classifier(
    model_variant: str,
    frontend_model: str,
    slot_vocab_sizes: tuple[int, ...] = SLOT_VOCAB_SIZES,
    feature_time_masking: bool = False,
    backbone_dropout: float = 0.3,
    head_dropout: float = 0.3,
    frontend_projection_dim: int = 256,
    slot_projection_dim: int = 256,
) -> LipReadingWordClassifier:
    """Factory for word-level GRID model variants."""
    return LipReadingWordClassifier(
        model_variant=model_variant,
        frontend_model=frontend_model,
        slot_vocab_sizes=slot_vocab_sizes,
        feature_time_masking=feature_time_masking,
        backbone_dropout=backbone_dropout,
        head_dropout=head_dropout,
        frontend_projection_dim=frontend_projection_dim,
        slot_projection_dim=slot_projection_dim,
    )


__all__ = [
    "LipReadingWordClassifier",
    "build_lipreading_word_classifier",
    "FRONTEND_MODELS",
    "MODEL_VARIANTS",
]
