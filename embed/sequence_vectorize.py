from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text

# TODO: Early tweets only have 140 charachters?
MAX_SEQUENCE_LENGTH = 140
TOP_K = 20000

def sequence_vectorize(texts):
    """Vectorizes texts as sequence vectors.
    # Arguments
        train_texts: list, training text strings.
    # Returns
        x_train, word_index: vectorized training and validation
            texts and word index dictionary.
    """
    # Create vocabulary with training texts.
    tokenizer = text.Tokenizer(num_words=TOP_K)
    tokenizer.fit_on_texts(texts)

    # Vectorize text.
    vectors = tokenizer.texts_to_sequences(texts)

    # Get max sequence length.
    max_length = len(max(vectors, key=len))
    if max_length > MAX_SEQUENCE_LENGTH:
        max_length = MAX_SEQUENCE_LENGTH

    # Add padding to sequences.
    padded_vectors = sequence.pad_sequences(vectors, maxlen=max_length)

    return padded_vectors, tokenizer.word_index