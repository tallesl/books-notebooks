import marimo

__generated_with = "0.8.1"
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
    input_filepath = 'principe.txt'

    with open(input_filepath, 'r', encoding='utf-8') as file:
        all_text = file.read()

    mo.md(f'''

    Loading the entire "{input_filepath}" file to memory:

    "{all_text[:200]}..."

    ''')
    return all_text, file, input_filepath


@app.cell
def __(all_text, keras, tf):
    batch_size = 32
    char_per_step = 100
    window_size = char_per_step + 1

    vectorizer = keras.layers.TextVectorization(standardize='lower_and_strip_punctuation', split='character')

    vectorizer.adapt(all_text)

    vocabulary = vectorizer.get_vocabulary()
    vocabulary_size = len(vocabulary)

    def split_input_and_target(windows):
        input = windows[:, :-1] # excludes the last token of the window, window size - 1 vector
        target = windows[:, 1:]  # excludes the first token of the window, windows size - 1 vector
        return (input, target)

    all_text_vectorized = vectorizer(all_text)

    dataset = tf.data.Dataset.from_tensor_slices(all_text_vectorized)

    dataset = dataset.window(window_size, shift=1, drop_remainder=True)

    dataset = dataset.shuffle(10000)

    dataset = dataset.flat_map(lambda window: window.batch(window_size))

    dataset = dataset.batch(batch_size)

    dataset = dataset.map(split_input_and_target)

    dataset = dataset.map(lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=vocabulary_size), Y_batch))

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return (
        all_text_vectorized,
        batch_size,
        char_per_step,
        dataset,
        split_input_and_target,
        vectorizer,
        vocabulary,
        vocabulary_size,
        window_size,
    )


@app.cell
def __(keras):
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath='epoch_{epoch:02d}_loss_{loss:.2f}.weights.h5',
        monitor='loss',
        save_best_only=True,
        save_weights_only=True,
        mode='min',
        verbose=1
    )
    return model_checkpoint,


@app.cell
def __(keras, vocabulary_size):
    early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

    model = keras.models.Sequential([
        keras.layers.GRU(128, return_sequences=True, input_shape=[None, vocabulary_size], dropout=0.2),
        keras.layers.GRU(128, return_sequences=True, dropout=0.2),
        keras.layers.TimeDistributed(keras.layers.Dense(vocabulary_size,
                                                        activation="softmax"))
    ])

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

    return early_stopping, model


@app.cell
def __(model):
    model.load_weights('epoch_05_loss_0.86.weights.h5')

    return


@app.cell
def __(dataset, early_stopping, model, model_checkpoint):
    history = model.fit(dataset, epochs=9999, callbacks=[model_checkpoint, early_stopping])
    return history,


@app.cell
def __(model, np, tf, vectorizer, vocabulary, vocabulary_size):
    def vector_to_string(vector):
        return ''.join(vocabulary[i] for i in vector)


    def vector_to_char(vector):
        return vocabulary[vector.numpy().item()]

    def preprocess(texts):
        X = np.array(vectorizer(texts))
        return tf.one_hot(X, vocabulary_size)

    def next_char(text, temperature=1):
        X_new = preprocess([text])
        logits = model(X_new)[0, -1, :] / temperature
        char_id = tf.random.categorical(logits[tf.newaxis, :], num_samples=1)
        return vector_to_char(char_id)

    def complete_text(text, n_chars=50, temperature=1):
        for _ in range(n_chars):
            text += next_char(text, temperature)
        return text
    return (
        complete_text,
        next_char,
        preprocess,
        vector_to_char,
        vector_to_string,
    )


@app.cell
def __(complete_text):
    beginning = 'princ'

    def preview(beginning, temperature):
        print(f'{temperature}: {complete_text(beginning, temperature=temperature)}')

    for _ in range(3):
        preview(beginning, 0.001)
        preview(beginning, 0.01)
        preview(beginning, 0.1)
        preview(beginning, 1)
        preview(beginning, 2)
        print()
    return beginning, preview


@app.cell
def __(model):
    model.summary()
    return


@app.cell
def __(vocabulary):
    vocabulary
    return


@app.cell
def __(model):
    model.layers[0].weights
    return


if __name__ == "__main__":
    app.run()
