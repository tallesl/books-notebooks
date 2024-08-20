import marimo

__generated_with = "0.8.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __(mo):
    mo.md("""# 1 - Initial setup""")
    return


@app.cell
def __(mo):
    import matplotlib
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras

    matplotlib_version = matplotlib.__version__
    numpy_version = np.version.version
    tensorflow_version = tf.__version__
    keras_version = keras.__version__

    mo.md(f'''

    Loading libraries:

    - Matplotlib version {matplotlib_version}
    - NumPy version: {numpy_version}
    - TensorFlow version: {tensorflow_version}
    - Keras version: {keras_version}

    ''')
    return (
        keras,
        keras_version,
        matplotlib,
        matplotlib_version,
        np,
        numpy_version,
        tensorflow_version,
        tf,
    )


@app.cell
def __(mo, np, tf):
    import matplotlib.pyplot as plt
    import random

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def set_memory_growth():
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
          tf.config.experimental.set_memory_growth(gpu, True)

    def plot_training(history):  
        plt.figure(figsize=(8, 5))  # 8x5 inches

        for key in history.keys():
            plt.plot(history[key], label=key)

        plt.grid(True)
        plt.ylim(0, 1)
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Learning Curve')
        plt.legend()
        plt.show()

    mo.md('''

    Defining utility functions.

    ''')
    return plot_training, plt, random, set_memory_growth, set_seed


@app.cell
def __(mo, set_memory_growth, set_seed):
    seed = 123

    set_seed(seed)
    set_memory_growth()

    mo.md('''

    Setting a seed and reducing GPU memory allocation.

    ''')
    return seed,


@app.cell
def __(mo):
    mo.md("""# 2 - Preparing the dataset""")
    return


@app.cell
def __(mo):
    input_filepath = 'harry-potter.txt'

    with open(input_filepath) as f:
        all_text = f.read()

    print(f'{all_text[:200]}...')

    mo.md(f'''

    Loading the entire "{input_filepath}" file to memory.

    Total number of characters: {len(all_text)}

    ''')
    return all_text, f, input_filepath


@app.cell
def __(all_text, mo, tf):
    vectorize_layer = tf.keras.layers.TextVectorization(
        split='character',
        standardize='lower_and_strip_punctuation'
    )

    vectorize_layer.adapt(all_text)

    vocabulary = vectorize_layer.get_vocabulary()
    vocabulary_size = len(vocabulary)

    mo.md(f'''

    Creating a [TextVectorization](https://keras.io/api/layers/preprocessing_layers/text/text_vectorization/) object with our loaded text. This will not only help now when loading the dataset but will also be used later as a preprocessing layer on our model.

    Vocabulary: {vocabulary}

    Vocabulary size: {vocabulary_size}

    ''')
    return vectorize_layer, vocabulary, vocabulary_size


@app.cell
def __(mo, vectorize_layer, vocabulary):
    def vector_to_text(vector):
        return ''.join(vocabulary[n] for n in vector)

    sample_text = "Hello world!"

    vectorized_text = vectorize_layer(sample_text)
    reconstructed_text = vector_to_text(vectorized_text)

    mo.md(f'''

    Converting "{sample_text}" to vector: {vectorized_text}

    Reconstructing the text from the vector: "{reconstructed_text}"

    ''')
    return reconstructed_text, sample_text, vector_to_text, vectorized_text


@app.cell
def __(all_text, tf):
    batch_size = 32
    characters_per_step = 100
    window_size = characters_per_step + 1

    # Step 1: Convert the entire text into a TensorFlow tensor
    text_tensor = tf.constant(all_text)

    # Step 2: Split the text tensor into individual characters using TensorFlow's built-in operations
    char_dataset = tf.data.Dataset.from_tensor_slices(tf.strings.unicode_split(text_tensor, 'UTF-8'))

    # Step 3: Build a StringLookup layer to map characters to integer indices
    lookup_layer = tf.keras.layers.StringLookup(vocabulary=list(set(all_text)), mask_token=None)

    def dataset_to_tensor(window):
        # Batch the elements in the window to create a single tensor of shape (window_size,)
        return window.batch(window_size)

    def split_input_target(window):
        # Ensure the tensor has at least rank 2 before slicing
        X = window[:, :-1]  # Take all except the last character for input
        Y = window[:, 1:]   # Take all except the first character for target
        return X, Y

    def one_hot_encode(X_batch, Y_batch):
        # Convert characters to integer indices
        X_batch = lookup_layer(X_batch)
        # One-hot encode the input characters using the vocabulary size
        depth = tf.cast(lookup_layer.vocabulary_size(), tf.int32)
        return tf.one_hot(X_batch, depth=depth), Y_batch

    dataset = (
        char_dataset
        .window(window_size, shift=1, drop_remainder=True)  # Create overlapping windows
        .flat_map(dataset_to_tensor)  # Convert windows to tensors
        .shuffle(10000)  # Shuffle the dataset
        .batch(batch_size)  # Batch the dataset
        .map(split_input_target)  # Split into input and target
        .map(one_hot_encode)  # Convert to indices and one-hot encode
        .prefetch(1)  # Prefetch for performance
    )

    # Debugging step: Print shapes of batches
    for X_batch, Y_batch in dataset.take(1):
        print("X_batch shape:", X_batch.shape)  # Should be (batch_size, characters_per_step, vocabulary_size)
        print("Y_batch shape:", Y_batch.shape)  # Should be (batch_size, characters_per_step)

    return (
        X_batch,
        Y_batch,
        batch_size,
        char_dataset,
        characters_per_step,
        dataset,
        dataset_to_tensor,
        lookup_layer,
        one_hot_encode,
        split_input_target,
        text_tensor,
        window_size,
    )


if __name__ == "__main__":
    app.run()
