import os
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.applications import efficientnet
from keras.layers import TextVectorization
from keras.utils import register_keras_serializable
from sklearn.model_selection import train_test_split

# Define constants
IMAGES_PATH = "/content/drive/MyDrive/Colab/flickr30k_images"
CAPTIONS_PATH = "/content/drive/MyDrive/Colab/results.csv"

# Desired image dimensions
IMAGE_SIZE = (299, 299)

# Fixed length allowed for any sequence
SEQ_LENGTH = 25

# Vocabulary size
VOCAB_SIZE = 10000

# Dimension for the image embeddings and token embeddings
EMBED_DIM = 512

# Number of units in the feed-forward network
FF_DIM = 512

# Number of attention heads
NUM_HEADS = 2

# Batch size
BATCH_SIZE = 256

# Number of epochs
EPOCHS = 30

def load_captions_data(filename):
    with open(filename, encoding='utf-8') as caption_file:
        caption_data = caption_file.readlines()[1:]
        caption_mapping = {}
        text_data = []
        images_to_skip = set()

        for line in caption_data:
            line = line.rstrip("\n")
            try:
                img_name, _, caption = line.split("| ")
            except ValueError:
                img_name, caption = line.split("| ")
                caption = caption[4:]

            img_name = os.path.join(IMAGES_PATH, img_name.strip())
            tokens = caption.strip().split()
            if len(tokens) < 4 or len(tokens) > SEQ_LENGTH:
                images_to_skip.add(img_name)
                continue

            if img_name.endswith("jpg") and img_name not in images_to_skip:
                caption = "<start> " + caption.strip() + " <end>"
                text_data.append(caption)

                if img_name in caption_mapping:
                    caption_mapping[img_name].append(caption)
                else:
                    caption_mapping[img_name] = [caption]

        for img_name in images_to_skip:
            if img_name in caption_mapping:
                del caption_mapping[img_name]

        return caption_mapping, text_data

def train_val_split(caption_data, validation_size=0.2, test_size=0.02, shuffle=True):
    all_images = list(caption_data.keys())
    if shuffle:
        np.random.shuffle(all_images)

    train_keys, validation_keys = train_test_split(
        all_images, test_size=validation_size, random_state=42
    )
    validation_keys, test_keys = train_test_split(
        validation_keys, test_size=test_size, random_state=42
    )

    training_data = {img_name: caption_data[img_name] for img_name in train_keys}
    validation_data = {img_name: caption_data[img_name] for img_name in validation_keys}
    test_data = {img_name: caption_data[img_name] for img_name in test_keys}

    return training_data, validation_data, test_data

# Load the dataset
captions_mapping, text_data = load_captions_data(CAPTIONS_PATH)

# Split the dataset
train_data, validation_data, test_data = train_val_split(captions_mapping)

print(f"Total samples: {len(captions_mapping)}")
print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(validation_data)}")
print(f"Test samples: {len(test_data)}")

def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    strip_chars = "!\"#$%&'()*+,-./:;=?@[\]^_`{|}~1234567890"
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

# Define the vectorizer
vectorization = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQ_LENGTH,
    standardize=custom_standardization
)

# Adapt the vectorizer to the text data
vectorization.adapt(text_data)

image_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomContrast(0.3)
])

@register_keras_serializable()
class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads

        # Initialize sub-layers in __init__
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.embed_dim
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.dense_1 = layers.Dense(self.embed_dim, activation="relu")

    def call(self, inputs, training, mask=None):
        inputs_norm = self.layernorm_1(inputs)
        inputs_dense = self.dense_1(inputs_norm)
        attention_output = self.attention_1(
            query=inputs_dense, value=inputs_dense, key=inputs_dense, training=training
        )
        out = self.layernorm_2(inputs_dense + attention_output)
        return out

    def get_config(self):
        config = super(TransformerEncoderBlock, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'dense_dim': self.dense_dim,
            'num_heads': self.num_heads,
        })
        return config

@register_keras_serializable()
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.token_embeddings = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=embed_dim)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # Remove 'embed_scale' from __init__

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embed_dim = tf.cast(self.embed_dim, tf.float32)
        embed_scale = tf.math.sqrt(embed_dim)
        embedded_tokens = self.token_embeddings(inputs) * embed_scale
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

    def get_config(self):
        config = super(PositionalEmbedding, self).get_config()
        config.update({
            'sequence_length': self.sequence_length,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
        })
        return config

@register_keras_serializable()
class TransformerDecoderBlock(layers.Layer):
    def __init__(self, embed_dim, ff_dim, num_heads, **kwargs):
        super(TransformerDecoderBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads

        # Initialize sub-layers in __init__
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.embed_dim
        )
        self.cross_attention_2 = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.embed_dim
        )
        self.ffn_layer_1 = layers.Dense(self.ff_dim, activation="relu")
        self.ffn_layer_2 = layers.Dense(self.embed_dim)
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.embedding = PositionalEmbedding(
            sequence_length=SEQ_LENGTH,
            vocab_size=VOCAB_SIZE,
            embed_dim=EMBED_DIM
        )
        self.out = layers.Dense(VOCAB_SIZE, activation="softmax")
        self.dropout_1 = layers.Dropout(0.3)
        self.dropout_2 = layers.Dropout(0.5)
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, training, mask=None):
        inputs_embedded = self.embedding(inputs)
        causal_mask = self.get_causal_attention_mask(inputs_embedded)

        if mask is not None:
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)
        else:
            combined_mask = causal_mask

        attention_output = self.attention_1(
            query=inputs_embedded,
            value=inputs_embedded,
            key=inputs_embedded,
            attention_mask=combined_mask,
            training=training
        )
        out1 = self.layernorm_1(inputs_embedded + attention_output)

        cross_attention_output = self.cross_attention_2(
            query=out1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask if mask is not None else None,
            training=training
        )
        out2 = self.layernorm_2(out1 + cross_attention_output)

        ffn_out = self.ffn_layer_1(out2)
        ffn_out = self.dropout_1(ffn_out, training=training)
        ffn_out = self.ffn_layer_2(ffn_out)

        ffn_out = self.layernorm_3(ffn_out + out2)
        ffn_out = self.dropout_2(ffn_out, training=training)

        preds = self.out(ffn_out)
        return preds

    def get_causal_attention_mask(self, inputs):
        batch_size, sequence_length = tf.shape(inputs)[0], tf.shape(inputs)[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, sequence_length, sequence_length))
        return tf.tile(mask, [batch_size, 1, 1])

    def get_config(self):
        config = super(TransformerDecoderBlock, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'ff_dim': self.ff_dim,
            'num_heads': self.num_heads,
        })
        return config

def get_cnn_model():
    base_model = efficientnet.EfficientNetB0(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False
    base_model_out = base_model.output
    base_model_out = layers.Reshape((-1, base_model_out.shape[-1]))(base_model_out)
    cnn_model = keras.models.Model(base_model.input, base_model_out)
    return cnn_model

cnn_model = get_cnn_model()

encoder = TransformerEncoderBlock(embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=NUM_HEADS)
decoder = TransformerDecoderBlock(embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=NUM_HEADS)

@register_keras_serializable()
class ImageCaptioningModel(keras.Model):
    def __init__(
        self,
        cnn_model,
        encoder,
        decoder,
        image_aug=None,
        **kwargs
    ):
        super(ImageCaptioningModel, self).__init__(**kwargs)
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.image_aug = image_aug

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.Mean(name="accuracy")

    def call(self, inputs, training=False, mask=None):
        batch_img, batch_seq = inputs

        if self.image_aug and training:
            batch_img = self.image_aug(batch_img)

        img_embed = self.cnn_model(batch_img, training=False)
        encoder_out = self.encoder(img_embed, training=training)

        batch_seq_inp = batch_seq[:, :-1]
        batch_seq_true = batch_seq[:, 1:]

        mask = tf.math.not_equal(batch_seq_inp, 0)

        batch_seq_pred = self.decoder(
            batch_seq_inp, encoder_out, training=training, mask=mask
        )

        return batch_seq_pred

    def calculate_loss(self, y_true, y_pred, mask):
        loss = self.compiled_loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def calculate_accuracy(self, y_true, y_pred, mask):
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
        accuracy = tf.math.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)

    def train_step(self, batch_data):
        batch_img, batch_seq = batch_data

        with tf.GradientTape() as tape:
            batch_seq_pred = self((batch_img, batch_seq), training=True)
            batch_seq_true = batch_seq[:, 1:]
            mask = tf.math.not_equal(batch_seq_true, 0)
            loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)

        train_vars = (
            self.encoder.trainable_variables +
            self.decoder.trainable_variables
        )
        gradients = tape.gradient(loss, train_vars)
        self.optimizer.apply_gradients(zip(gradients, train_vars))

        acc = self.calculate_accuracy(batch_seq_true, batch_seq_pred, mask)

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)

        return {"loss": self.loss_tracker.result(), "accuracy": self.acc_tracker.result()}

    def test_step(self, batch_data):
        batch_img, batch_seq = batch_data

        batch_seq_pred = self((batch_img, batch_seq), training=False)
        batch_seq_true = batch_seq[:, 1:]
        mask = tf.math.not_equal(batch_seq_true, 0)
        loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
        acc = self.calculate_accuracy(batch_seq_true, batch_seq_pred, mask)

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)

        return {"loss": self.loss_tracker.result(), "accuracy": self.acc_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]

    def get_config(self):
        config = super(ImageCaptioningModel, self).get_config()
        config.update({
            'cnn_model': keras.layers.serialize(self.cnn_model),
            'encoder': keras.layers.serialize(self.encoder),
            'decoder': keras.layers.serialize(self.decoder),
            'image_aug': keras.layers.serialize(self.image_aug) if self.image_aug else None,
        })
        return config

    @classmethod
    def from_config(cls, config):
        cnn_model_config = config.pop('cnn_model')
        encoder_config = config.pop('encoder')
        decoder_config = config.pop('decoder')
        image_aug_config = config.pop('image_aug', None)

        cnn_model = keras.layers.deserialize(cnn_model_config, custom_objects={'EfficientNetB0': efficientnet.EfficientNetB0})
        encoder = keras.layers.deserialize(encoder_config, custom_objects={'TransformerEncoderBlock': TransformerEncoderBlock})
        decoder = keras.layers.deserialize(decoder_config, custom_objects={
            'TransformerDecoderBlock': TransformerDecoderBlock,
            'PositionalEmbedding': PositionalEmbedding
        })
        image_aug = keras.layers.deserialize(image_aug_config) if image_aug_config else None

        return cls(
            cnn_model=cnn_model,
            encoder=encoder,
            decoder=decoder,
            image_aug=image_aug,
            **config
        )

def decode_and_resize(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def process_input(img_path, caption):
    img = decode_and_resize(img_path)
    caption = vectorization(caption)
    return img, caption

def flatten_data(images, captions):
    flattened_images = []
    flattened_captions = []
    for img, caps in zip(images, captions):
        for cap in caps:
            flattened_images.append(img)
            flattened_captions.append(cap)
    return flattened_images, flattened_captions

def make_dataset(images, captions):
    images, captions = flatten_data(images, captions)
    dataset = tf.data.Dataset.from_tensor_slices((images, captions))
    dataset = dataset.map(process_input, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(BATCH_SIZE * 8).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset

# Prepare the datasets
train_dataset = make_dataset(list(train_data.keys()), list(train_data.values()))
validation_dataset = make_dataset(list(validation_data.keys()), list(validation_data.values()))

# Define the loss function
cross_entropy = keras.losses.SparseCategoricalCrossentropy(
    from_logits=False, reduction='none'
)