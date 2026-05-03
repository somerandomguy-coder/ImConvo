"""
CTC-based lip reading model (TensorFlow/Keras).

Architecture: 3D-CNN front-end -> temporal backbone -> per-frame character logits.
Supported temporal backbones:
- bigru (baseline)
- gru (single-direction)
- bilstm
- transformer (small)
- tcn
- conformer_lite
- transformer_medium
"""

from __future__ import annotations

import tensorflow as tf
from keras import Model, layers
from src.utils import BLANK_IDX, MAX_FRAMES, NUM_CHARS

MODEL_VARIANTS = (
    "bigru",
    "gru",
    "bilstm",
    "transformer",
    "tcn",
    "conformer_lite",
    "transformer_medium",
)


class TransformerEncoderBlock(layers.Layer):
    """Small transformer encoder block for temporal modeling."""

    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float, **kwargs):
        super().__init__(**kwargs)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.dropout1 = layers.Dropout(dropout)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)

        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dropout(dropout),
                layers.Dense(d_model),
            ]
        )
        self.dropout2 = layers.Dropout(dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=False):
        attn_out = self.attn(x, x, training=training)
        attn_out = self.dropout1(attn_out, training=training)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x, training=training)
        ffn_out = self.dropout2(ffn_out, training=training)
        return self.norm2(x + ffn_out)


class TemporalConvBlock(layers.Layer):
    """Residual dilated temporal Conv1D block for TCN backbone."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation_rate: int,
        dropout: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.conv = layers.Conv1D(
            filters=channels,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="causal",
        )
        self.norm = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.dropout = layers.Dropout(dropout)
        self.residual_proj = None

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        conv_dim = int(self.conv.filters)
        if input_dim != conv_dim:
            self.residual_proj = layers.Dense(conv_dim)
        super().build(input_shape)

    def call(self, x, training=False):
        residual = x if self.residual_proj is None else self.residual_proj(x)
        out = self.conv(x)
        out = self.norm(out, training=training)
        out = self.relu(out)
        out = self.dropout(out, training=training)
        return residual + out


class ConformerBlock(layers.Layer):
    """Conformer-lite block with FFN + MHSA + convolution module."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_multiplier: int,
        conv_kernel_size: int,
        dropout: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        ff_dim = d_model * ff_multiplier
        self.ffn1_norm = layers.LayerNormalization(epsilon=1e-6)
        self.ffn1_dense1 = layers.Dense(ff_dim, activation="swish")
        self.ffn1_drop1 = layers.Dropout(dropout)
        self.ffn1_dense2 = layers.Dense(d_model)
        self.ffn1_drop2 = layers.Dropout(dropout)

        self.attn_norm = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
        )
        self.attn_drop = layers.Dropout(dropout)

        self.conv_norm = layers.LayerNormalization(epsilon=1e-6)
        self.conv_pw1 = layers.Conv1D(2 * d_model, kernel_size=1, padding="same")
        self.conv_dw = layers.SeparableConv1D(
            filters=d_model,
            kernel_size=conv_kernel_size,
            padding="same",
        )
        self.conv_bn = layers.BatchNormalization()
        self.conv_act = layers.Activation("swish")
        self.conv_pw2 = layers.Conv1D(d_model, kernel_size=1, padding="same")
        self.conv_drop = layers.Dropout(dropout)

        self.ffn2_norm = layers.LayerNormalization(epsilon=1e-6)
        self.ffn2_dense1 = layers.Dense(ff_dim, activation="swish")
        self.ffn2_drop1 = layers.Dropout(dropout)
        self.ffn2_dense2 = layers.Dense(d_model)
        self.ffn2_drop2 = layers.Dropout(dropout)
        self.final_norm = layers.LayerNormalization(epsilon=1e-6)

    def _ffn(self, x, dense1, drop1, dense2, drop2, training=False):
        out = dense1(x)
        out = drop1(out, training=training)
        out = dense2(out)
        out = drop2(out, training=training)
        return out

    def call(self, x, training=False):
        x = x + 0.5 * self._ffn(
            self.ffn1_norm(x),
            self.ffn1_dense1,
            self.ffn1_drop1,
            self.ffn1_dense2,
            self.ffn1_drop2,
            training=training,
        )

        attn_out = self.attn(self.attn_norm(x), self.attn_norm(x), training=training)
        attn_out = self.attn_drop(attn_out, training=training)
        x = x + attn_out

        conv_x = self.conv_norm(x)
        conv_x = self.conv_pw1(conv_x)
        conv_left, conv_right = tf.split(conv_x, num_or_size_splits=2, axis=-1)
        conv_x = conv_left * tf.nn.sigmoid(conv_right)
        conv_x = self.conv_dw(conv_x)
        conv_x = self.conv_bn(conv_x, training=training)
        conv_x = self.conv_act(conv_x)
        conv_x = self.conv_pw2(conv_x)
        conv_x = self.conv_drop(conv_x, training=training)
        x = x + conv_x

        x = x + 0.5 * self._ffn(
            self.ffn2_norm(x),
            self.ffn2_dense1,
            self.ffn2_drop1,
            self.ffn2_dense2,
            self.ffn2_drop2,
            training=training,
        )
        return self.final_norm(x)


class LipReadingCTC(Model):
    """
    3D-CNN + configurable temporal backbone lip reading model with CTC output.

    Input:  (batch, T=75, H=80, W=120, 1)
    Output: (batch, T=75, num_chars)
    """

    def __init__(
        self,
        num_chars: int = NUM_CHARS,
        model_variant: str = "bigru",
        feature_time_masking: bool = False,
        **kwargs,
    ):
        super().__init__(name="LipReadingCTC", **kwargs)
        model_variant = model_variant.lower()
        if model_variant not in MODEL_VARIANTS:
            raise ValueError(
                f"Unsupported model_variant='{model_variant}'. "
                f"Supported: {MODEL_VARIANTS}"
            )

        self.num_chars = num_chars
        self.model_variant = model_variant
        self.feature_time_masking = bool(feature_time_masking)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        # 3D CNN front-end
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

        # Temporal backbone blocks (only one path used in call)
        self.temporal_bigru1 = layers.Bidirectional(
            layers.GRU(256, return_sequences=True, dropout=0.3),
            name="temporal_bigru1",
        )
        self.temporal_bigru2 = layers.Bidirectional(
            layers.GRU(256, return_sequences=True, dropout=0.3),
            name="temporal_bigru2",
        )

        self.temporal_gru1 = layers.GRU(
            512, return_sequences=True, dropout=0.3, name="temporal_gru1"
        )
        self.temporal_gru2 = layers.GRU(
            512, return_sequences=True, dropout=0.3, name="temporal_gru2"
        )

        self.temporal_bilstm1 = layers.Bidirectional(
            layers.LSTM(256, return_sequences=True, dropout=0.3),
            name="temporal_bilstm1",
        )
        self.temporal_bilstm2 = layers.Bidirectional(
            layers.LSTM(256, return_sequences=True, dropout=0.3),
            name="temporal_bilstm2",
        )

        # Transformer (small baseline)
        self.transformer_proj = layers.Dense(256, name="transformer_proj")
        self.transformer_pos_embed = layers.Embedding(MAX_FRAMES, 256, name="transformer_pos_embed")
        self.transformer_drop = layers.Dropout(0.3)
        self.transformer_block1 = TransformerEncoderBlock(
            d_model=256,
            num_heads=4,
            ff_dim=512,
            dropout=0.3,
            name="transformer_block1",
        )
        self.transformer_block2 = TransformerEncoderBlock(
            d_model=256,
            num_heads=4,
            ff_dim=512,
            dropout=0.3,
            name="transformer_block2",
        )

        # Temporal CNN / TCN
        self.tcn_proj = layers.Dense(384, name="tcn_proj")
        self.tcn_blocks = [
            TemporalConvBlock(
                channels=384,
                kernel_size=5,
                dilation_rate=dilation,
                dropout=0.3,
                name=f"tcn_block_d{dilation}",
            )
            for dilation in (1, 2, 4, 8)
        ]

        # Conformer-lite
        self.conformer_proj = layers.Dense(256, name="conformer_proj")
        self.conformer_pos_embed = layers.Embedding(
            MAX_FRAMES,
            256,
            name="conformer_pos_embed",
        )
        self.conformer_drop = layers.Dropout(0.3)
        self.conformer_blocks = [
            ConformerBlock(
                d_model=256,
                num_heads=4,
                ff_multiplier=4,
                conv_kernel_size=15,
                dropout=0.3,
                name=f"conformer_block{i + 1}",
            )
            for i in range(3)
        ]

        # Transformer-medium
        self.transformer_medium_proj = layers.Dense(384, name="transformer_medium_proj")
        self.transformer_medium_pos_embed = layers.Embedding(
            MAX_FRAMES,
            384,
            name="transformer_medium_pos_embed",
        )
        self.transformer_medium_drop = layers.Dropout(0.3)
        self.transformer_medium_blocks = [
            TransformerEncoderBlock(
                d_model=384,
                num_heads=4,
                ff_dim=768,
                dropout=0.3,
                name=f"transformer_medium_block{i + 1}",
            )
            for i in range(4)
        ]

        # Per-frame character output
        self.char_dense = layers.Dense(128, activation="relu")
        self.char_dropout = layers.Dropout(0.3)
        self.char_output = layers.Dense(num_chars, activation=None, name="char_logits")

    def _add_positional_embedding(
        self,
        x: tf.Tensor,
        pos_embed: layers.Embedding,
    ) -> tf.Tensor:
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        pos = pos_embed(positions)
        return x + pos[tf.newaxis, :, :]

    def _apply_temporal_backbone(self, x, training=False):
        if self.model_variant == "bigru":
            x = self.temporal_bigru1(x, training=training)
            x = self.temporal_bigru2(x, training=training)
            return x
        if self.model_variant == "gru":
            x = self.temporal_gru1(x, training=training)
            x = self.temporal_gru2(x, training=training)
            return x
        if self.model_variant == "bilstm":
            x = self.temporal_bilstm1(x, training=training)
            x = self.temporal_bilstm2(x, training=training)
            return x
        if self.model_variant == "tcn":
            x = self.tcn_proj(x)
            for block in self.tcn_blocks:
                x = block(x, training=training)
            return x
        if self.model_variant == "conformer_lite":
            x = self.conformer_proj(x)
            x = self._add_positional_embedding(x, self.conformer_pos_embed)
            x = self.conformer_drop(x, training=training)
            for block in self.conformer_blocks:
                x = block(x, training=training)
            return x
        if self.model_variant == "transformer_medium":
            x = self.transformer_medium_proj(x)
            x = self._add_positional_embedding(x, self.transformer_medium_pos_embed)
            x = self.transformer_medium_drop(x, training=training)
            for block in self.transformer_medium_blocks:
                x = block(x, training=training)
            return x

        # transformer
        x = self.transformer_proj(x)
        x = self._add_positional_embedding(x, self.transformer_pos_embed)
        x = self.transformer_drop(x, training=training)
        x = self.transformer_block1(x, training=training)
        x = self.transformer_block2(x, training=training)
        return x

    def get_frontend_layers(self) -> list[layers.Layer]:
        """Return layers that form the visual frontend stack."""
        return [
            self.conv1,
            self.bn1,
            self.relu1,
            self.pool1,
            self.conv2,
            self.bn2,
            self.relu2,
            self.pool2,
            self.conv3,
            self.bn3,
            self.relu3,
            self.pool3,
            self.time_flatten,
        ]

    def get_backbone_layers(self) -> list[layers.Layer]:
        """Return temporal backbone layers for all variants."""
        return [
            self.temporal_bigru1,
            self.temporal_bigru2,
            self.temporal_gru1,
            self.temporal_gru2,
            self.temporal_bilstm1,
            self.temporal_bilstm2,
            self.transformer_proj,
            self.transformer_pos_embed,
            self.transformer_drop,
            self.transformer_block1,
            self.transformer_block2,
            self.tcn_proj,
            *self.tcn_blocks,
            self.conformer_proj,
            self.conformer_pos_embed,
            self.conformer_drop,
            *self.conformer_blocks,
            self.transformer_medium_proj,
            self.transformer_medium_pos_embed,
            self.transformer_medium_drop,
            *self.transformer_medium_blocks,
        ]

    def get_head_layers(self) -> list[layers.Layer]:
        """Return classification head layers."""
        return [
            self.char_dense,
            self.char_dropout,
            self.char_output,
        ]

    def _apply_feature_time_mask(self, x: tf.Tensor) -> tf.Tensor:
        """Mask 2-6 contiguous timesteps across all feature dimensions."""
        batch_size = tf.shape(x)[0]
        mask_len = tf.random.uniform([batch_size], minval=2, maxval=7, dtype=tf.int32)
        max_start = MAX_FRAMES - mask_len + 1
        start = tf.cast(
            tf.random.uniform([batch_size], 0.0, 1.0) * tf.cast(max_start, tf.float32),
            tf.int32,
        )
        time_idx = tf.range(MAX_FRAMES)[tf.newaxis, :]
        mask = tf.logical_and(
            time_idx >= start[:, tf.newaxis],
            time_idx < (start + mask_len)[:, tf.newaxis],
        )
        return tf.where(mask[:, :, tf.newaxis], tf.zeros_like(x), x)

    def call(self, inputs, training=False):
        x = self.pool1(self.relu1(self.bn1(self.conv1(inputs), training=training)))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x), training=training)))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x), training=training)))

        x = self.time_flatten(x)
        if self.feature_time_masking and training:
            x = tf.cond(
                tf.random.uniform([], 0.0, 1.0) < 0.5,
                lambda: self._apply_feature_time_mask(x),
                lambda: x,
            )
        x = self._apply_temporal_backbone(x, training=training)

        x = self.char_dense(x)
        x = self.char_dropout(x, training=training)
        return self.char_output(x)

    def _compute_ctc_loss(self, y_pred, labels, label_length):
        batch_size = tf.shape(y_pred)[0]
        input_length = tf.fill([batch_size], tf.cast(MAX_FRAMES, tf.int32))
        label_length = tf.cast(tf.reshape(label_length, [-1]), tf.int32)

        logits_time_major = tf.transpose(y_pred, [1, 0, 2])

        loss = tf.nn.ctc_loss(
            labels=tf.cast(labels, tf.int32),
            logits=logits_time_major,
            label_length=label_length,
            logit_length=input_length,
            logits_time_major=True,
            blank_index=BLANK_IDX,
        )
        return tf.reduce_mean(loss)

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        x, y = data
        labels = y["labels"]
        label_length = y["label_length"]

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self._compute_ctc_loss(y_pred, labels, label_length)

        grads = tape.gradient(loss, self.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        x, y = data
        labels = y["labels"]
        label_length = y["label_length"]

        y_pred = self(x, training=False)
        loss = self._compute_ctc_loss(y_pred, labels, label_length)

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def decode_greedy(self, y_pred):
        logits_t = tf.transpose(y_pred, [1, 0, 2])
        input_length = tf.fill([tf.shape(y_pred)[0]], MAX_FRAMES)

        decoded, _ = tf.nn.ctc_greedy_decoder(
            logits_t, input_length, blank_index=BLANK_IDX
        )
        sparse = decoded[0]
        dense = tf.sparse.to_dense(sparse, default_value=-1)
        return dense.numpy()

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_chars": self.num_chars,
                "model_variant": self.model_variant,
                "feature_time_masking": self.feature_time_masking,
            }
        )
        return config


def build_lipreading_ctc(
    model_variant: str,
    num_chars: int = NUM_CHARS,
    feature_time_masking: bool = False,
) -> LipReadingCTC:
    """Factory for lipreading CTC model variants."""
    return LipReadingCTC(
        num_chars=num_chars,
        model_variant=model_variant,
        feature_time_masking=feature_time_masking,
    )


class LegacyLipReadingCTC(Model):
    """
    Backward-compatible model for old checkpoints that used bigru1/bigru2 naming.
    Intended only for loading legacy inference checkpoints.
    """

    def __init__(self, num_chars: int = NUM_CHARS, **kwargs):
        super().__init__(name="LegacyLipReadingCTC", **kwargs)
        self.num_chars = num_chars

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

        # Legacy names:
        self.bigru1 = layers.Bidirectional(
            layers.GRU(256, return_sequences=True, dropout=0.3)
        )
        self.bigru2 = layers.Bidirectional(
            layers.GRU(256, return_sequences=True, dropout=0.3)
        )

        self.char_dense = layers.Dense(128, activation="relu")
        self.char_dropout = layers.Dropout(0.3)
        self.char_output = layers.Dense(num_chars, activation=None, name="char_logits")

    def call(self, inputs, training=False):
        x = self.pool1(self.relu1(self.bn1(self.conv1(inputs), training=training)))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x), training=training)))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x), training=training)))
        x = self.time_flatten(x)
        x = self.bigru1(x, training=training)
        x = self.bigru2(x, training=training)
        x = self.char_dense(x)
        x = self.char_dropout(x, training=training)
        return self.char_output(x)

    def decode_greedy(self, y_pred):
        logits_t = tf.transpose(y_pred, [1, 0, 2])
        input_length = tf.fill([tf.shape(y_pred)[0]], MAX_FRAMES)
        decoded, _ = tf.nn.ctc_greedy_decoder(
            logits_t, input_length, blank_index=BLANK_IDX
        )
        sparse = decoded[0]
        dense = tf.sparse.to_dense(sparse, default_value=-1)
        return dense.numpy()


def count_parameters(model: Model) -> int:
    """Count trainable parameters."""
    return int(sum(tf.reduce_prod(w.shape) for w in model.trainable_weights))


if __name__ == "__main__":
    import numpy as np

    for variant in MODEL_VARIANTS:
        model = build_lipreading_ctc(model_variant=variant, num_chars=NUM_CHARS)
        dummy = np.random.randn(2, 75, 80, 120, 1).astype(np.float32)
        logits = model(dummy)
        print(f"{variant}: output={logits.shape}, params={count_parameters(model):,}")
