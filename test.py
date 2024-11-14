# Load the saved vocabulary
import pickle
with open('/content/drive/MyDrive/Colab/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

# Recreate the TextVectorization layer and set the vocabulary
vectorization = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQ_LENGTH,
    standardize=custom_standardization
)
vectorization.set_vocabulary(vocab)

# Define custom objects
custom_objects = {
    'ImageCaptioningModel': ImageCaptioningModel,
    'TransformerEncoderBlock': TransformerEncoderBlock,
    'TransformerDecoderBlock': TransformerDecoderBlock,
    'PositionalEmbedding': PositionalEmbedding
}

# Load the model
loaded_model = keras.models.load_model('/content/drive/MyDrive/Colab/caption_model_test.keras', custom_objects=custom_objects)

MAX_DECODED_SENTENCE_LENGTH = SEQ_LENGTH - 1
INDEX_TO_WORD = {idx: word for idx, word in enumerate(vocab)}

def greedy_algorithm(image):
    # Read the image from the disk
    image = decode_and_resize(image)

    # Pass the image to the CNN
    image = tf.expand_dims(image, 0)
    image = loaded_model.cnn_model(image)

    # Pass the image features to the Transformer encoder
    encoded_img = loaded_model.encoder(image, training=False)

    # Generate the caption using the Transformer decoder
    decoded_caption = "<start>"
    for i in range(MAX_DECODED_SENTENCE_LENGTH):
        tokenized_caption = vectorization([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = loaded_model.decoder(tokenized_caption, encoded_img, training=False, mask=mask)
        sampled_token_index = np.argmax(predictions[0, i, :])

        # Use the consistent INDEX_TO_WORD mapping
        sampled_token = INDEX_TO_WORD.get(sampled_token_index, "<unk>")

        if sampled_token == "<end>":
            break
        decoded_caption += " " + sampled_token

    decoded_caption = decoded_caption.replace("<start> ", "").strip()

    return decoded_caption

from PIL import Image
img = "/content/drive/MyDrive/Colab/000000000785.jpg"
caption = greedy_algorithm(img)
print(f'Generated Caption: {caption}\n')
Image.open(img)