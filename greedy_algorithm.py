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