import marimo

__generated_with = "0.8.0"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __():
    import numpy as np
    def str_array_fix(a):
        return np.array(a).astype(object)
    return np, str_array_fix


@app.cell
def __(mo):
    mo.md(
        r"""
        **Chapter 16 – Natural Language Processing with RNNs and Attention**

        _This notebook contains all the sample code in chapter 16._

        # Setup

        First, let's import a few common modules, ensure MatplotLib plots figures inline and prepare a function to save the figures. We also check that Python 3.5 or later is installed (although Python 2.x may work, it is deprecated so we strongly recommend you use Python 3 instead), as well as Scikit-Learn ≥0.20 and TensorFlow ≥2.0.
        """
    )
    return


@app.cell
def __(IS_COLAB, IS_KAGGLE, np):
    # Python ≥3.5 is required
    import sys
    assert sys.version_info >= (3, 5)

    # TensorFlow ≥2.0 is required
    import tensorflow as tf
    from tensorflow import keras
    assert tf.__version__ >= "2.0"

    if not tf.config.list_physical_devices('GPU'):
        print("No GPU was detected. LSTMs and CNNs can be very slow without a GPU.")
        if IS_COLAB:
            print("Go to Runtime > Change runtime and select a GPU hardware accelerator.")
        if IS_KAGGLE:
            print("Go to Settings > Accelerator and select GPU.")

    # Common imports
    import os

    # to make this notebook's output stable across runs
    np.random.seed(42)
    tf.random.set_seed(42)

    # To plot pretty figures
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rc('axes', labelsize=14)
    mpl.rc('xtick', labelsize=12)
    mpl.rc('ytick', labelsize=12)

    # Where to save the figures
    PROJECT_ROOT_DIR = "."
    CHAPTER_ID = "nlp"
    IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
    os.makedirs(IMAGES_PATH, exist_ok=True)

    def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
        path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
        print("Saving figure", fig_id)
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution)
    return (
        CHAPTER_ID,
        IMAGES_PATH,
        PROJECT_ROOT_DIR,
        keras,
        mpl,
        os,
        plt,
        save_fig,
        sys,
        tf,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        # Char-RNN

        ## Splitting a sequence into batches of shuffled windows

        For example, let's split the sequence 0 to 14 into windows of length 5, each shifted by 2 (e.g.,`[0, 1, 2, 3, 4]`, `[2, 3, 4, 5, 6]`, etc.), then shuffle them, and split them into inputs (the first 4 steps) and targets (the last 4 steps) (e.g., `[2, 3, 4, 5, 6]` would be split into `[[2, 3, 4, 5], [3, 4, 5, 6]]`), then create batches of 3 such input/target pairs:
        """
    )
    return


@app.cell
def __(np, tf):
    np.random.seed(42)
    tf.random.set_seed(42)

    def preview_dataset():
        n_steps = 5
        dataset = tf.data.Dataset.from_tensor_slices(tf.range(15))
        dataset = dataset.window(n_steps, shift=2, drop_remainder=True)
        dataset = dataset.flat_map(lambda window: window.batch(n_steps))
        dataset = dataset.shuffle(10).map(lambda window: (window[:-1], window[1:]))
        dataset = dataset.batch(3).prefetch(1)
        for index, (X_batch, Y_batch) in enumerate(dataset):
            print("_" * 20, "Batch", index, "\nX_batch")
            print(X_batch.numpy())
            print("=" * 5, "\nY_batch")
            print(Y_batch.numpy())

    preview_dataset()
    return preview_dataset,


@app.cell
def __(mo):
    mo.md(r"""## Loading the Data and Preparing the Dataset""")
    return


@app.cell
def __():
    with open('shakespeare.txt') as f:
        shakespeare_text = f.read()
    return f, shakespeare_text


@app.cell
def __(shakespeare_text):
    print(shakespeare_text[:148])
    return


@app.cell
def __(shakespeare_text):
    "".join(sorted(set(shakespeare_text.lower())))
    return


@app.cell
def __(keras, shakespeare_text):
    tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
    tokenizer.fit_on_texts(shakespeare_text)
    return tokenizer,


@app.cell
def __(tokenizer):
    tokenizer.texts_to_sequences(["First"])
    return


@app.cell
def __(tokenizer):
    tokenizer.sequences_to_texts([[20, 6, 9, 8, 3]])
    return


@app.cell
def __(tokenizer):
    max_id = len(tokenizer.word_index) # number of distinct characters
    dataset_size = tokenizer.document_count # total number of characters
    return dataset_size, max_id


@app.cell
def __(dataset_size, max_id, np, shakespeare_text, tf, tokenizer):
    [encoded] = np.array(tokenizer.texts_to_sequences([shakespeare_text])) - 1
    train_size = dataset_size * 90 // 100
    dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])


    n_steps = 100
    window_length = n_steps + 1 # target = input shifted 1 character ahead
    dataset = dataset.window(window_length, shift=1, drop_remainder=True)


    dataset = dataset.flat_map(lambda window: window.batch(window_length))


    np.random.seed(42)
    tf.random.set_seed(42)


    batch_size = 32
    dataset = dataset.shuffle(10000).batch(batch_size)
    dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))


    dataset = dataset.map(
        lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))


    dataset = dataset.prefetch(1)
    return batch_size, dataset, encoded, n_steps, train_size, window_length


@app.cell
def __(dataset):
    for X_batch, Y_batch in dataset.take(1):
        print(X_batch.shape, Y_batch.shape)
    return X_batch, Y_batch


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Creating and Training the Model

        **Warning**: the following code may take up to 24 hours to run, depending on your hardware. If you use a GPU, it may take just 1 or 2 hours, or less.

        **Note**: the `GRU` class will only use the GPU (if you have one) when using the default values for the following arguments: `activation`, `recurrent_activation`, `recurrent_dropout`, `unroll`, `use_bias` and `reset_after`. This is why I commented out `recurrent_dropout=0.2` (compared to the book).
        """
    )
    return


@app.cell
def __(dataset, keras, max_id):
    model = keras.models.Sequential([
        keras.layers.GRU(128, return_sequences=True, input_shape=[None, max_id],
                         #dropout=0.2, recurrent_dropout=0.2),
                         dropout=0.2),
        keras.layers.GRU(128, return_sequences=True,
                         #dropout=0.2, recurrent_dropout=0.2),
                         dropout=0.2),
        keras.layers.TimeDistributed(keras.layers.Dense(max_id,
                                                        activation="softmax"))
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    history = model.fit(dataset, epochs=2)
    return history, model


@app.cell
def __(mo):
    mo.md(r"""## Using the Model to Generate Text""")
    return


@app.cell
def __(max_id, np, tf, tokenizer):
    def preprocess(texts):
        X = np.array(tokenizer.texts_to_sequences(texts)) - 1
        return tf.one_hot(X, max_id)
    return preprocess,


@app.cell
def __(mo):
    mo.md(r"""**Warning**: the `predict_classes()` method is deprecated. Instead, we must use `np.argmax(model(X_new), axis=-1)`.""")
    return


@app.cell
def __(model, np, preprocess, tokenizer):
    X_new = preprocess(["How are yo"])
    #Y_pred = model.predict_classes(X_new)
    Y_pred = np.argmax(model(X_new), axis=-1)
    tokenizer.sequences_to_texts(Y_pred + 1)[0][-1] # 1st sentence, last char
    return X_new, Y_pred


@app.cell
def __(np, tf):
    tf.random.set_seed(42)

    tf.random.categorical([[np.log(0.5), np.log(0.4), np.log(0.1)]], num_samples=40).numpy()
    return


@app.cell
def __(model, preprocess, tf, tokenizer):
    def next_char(text, temperature=1):
        X_new = preprocess([text])
        y_proba = model(X_new)[0, -1:, :]
        rescaled_logits = tf.math.log(y_proba) / temperature
        char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1
        return tokenizer.sequences_to_texts(char_id.numpy())[0]
    return next_char,


@app.cell
def __(next_char, tf):
    tf.random.set_seed(42)

    next_char("How are yo", temperature=1)
    return


@app.cell
def __(next_char):
    def complete_text(text, n_chars=50, temperature=1):
        for _ in range(n_chars):
            text += next_char(text, temperature)
        return text
    return complete_text,


@app.cell
def __(complete_text, tf):
    tf.random.set_seed(42)

    print(complete_text("t", temperature=0.2))
    return


@app.cell
def __(complete_text):
    print(complete_text("t", temperature=1))
    return


@app.cell
def __(complete_text):
    print(complete_text("t", temperature=2))
    return


@app.cell
def __(mo):
    mo.md(r"""## Stateful RNN""")
    return


@app.cell
def __(tf):
    tf.random.set_seed(42)
    return


@app.cell
def __():
    #dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])
    #dataset = dataset.window(window_length, shift=n_steps, drop_remainder=True)
    #dataset = dataset.flat_map(lambda window: window.batch(window_length))
    #dataset = dataset.batch(1)
    #dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
    #dataset = dataset.map(
    #    lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))
    #dataset = dataset.prefetch(1)
    return


@app.cell
def __():
    #batch_size = 32
    #encoded_parts = np.array_split(encoded[:train_size], batch_size)
    #datasets = []
    #for encoded_part in encoded_parts:
    #    dataset = tf.data.Dataset.from_tensor_slices(encoded_part)
    #    dataset = dataset.window(window_length, shift=n_steps, drop_remainder=True)
    #    dataset = dataset.flat_map(lambda window: window.batch(window_length))
    #    datasets.append(dataset)
    #dataset = tf.data.Dataset.zip(tuple(datasets)).map(lambda *windows: tf.stack(windows))
    #dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
    #dataset = dataset.map(
    #    lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))
    #dataset = dataset.prefetch(1)
    return


@app.cell
def __(mo):
    mo.md(r"""**Note**: once again, I commented out `recurrent_dropout=0.2` (compared to the book) so you can get GPU acceleration (if you have one).""")
    return


@app.cell
def __(keras, max_id):
    stateful_model = keras.models.Sequential([
        keras.layers.GRU(128, return_sequences=True, stateful=True,
                         #dropout=0.2, recurrent_dropout=0.2,
                         dropout=0.2),
        keras.layers.GRU(128, return_sequences=True, stateful=True,
                         #dropout=0.2, recurrent_dropout=0.2),
                         dropout=0.2),
        keras.layers.TimeDistributed(keras.layers.Dense(max_id,
                                                        activation="softmax"))
    ])
    return stateful_model,


@app.cell
def __(keras):
    class ResetStatesCallback(keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            for layer in self.model.layers:
                if hasattr(layer, 'reset_states'):
                    layer.reset_states()
    return ResetStatesCallback,


@app.cell
def __(ResetStatesCallback, dataset, model):
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    stateful_history = model.fit(dataset, epochs=20,
                        callbacks=[ResetStatesCallback()])
    return stateful_history,


@app.cell
def __(mo):
    mo.md(r"""To use the model with different batch sizes, we need to create a stateless copy. We can get rid of dropout since it is only used during training:""")
    return


@app.cell
def __(keras, max_id):
    stateless_model = keras.models.Sequential([
        keras.layers.GRU(128, return_sequences=True, input_shape=[None, max_id]),
        keras.layers.GRU(128, return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(max_id,
                                                        activation="softmax"))
    ])
    return stateless_model,


@app.cell
def __(mo):
    mo.md(r"""To set the weights, we first need to build the model (so the weights get created):""")
    return


@app.cell
def __(max_id, stateless_model, tf):
    stateless_model.build(tf.TensorShape([None, None, max_id]))
    return


@app.cell
def __(model, stateless_model):
    stateless_model.set_weights(model.get_weights())
    return


@app.cell
def __(preprocess, stateless_model, tf, tokenizer):
    tf.random.set_seed(42)

    def next_char2(text, temperature=1):
        X_new = preprocess([text])
        y_proba = stateless_model(X_new)[0, -1:, :]
        rescaled_logits = tf.math.log(y_proba) / temperature
        char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1
        return tokenizer.sequences_to_texts(char_id.numpy())[0]

    def complete_text2(text, n_chars=50, temperature=1):
        for _ in range(n_chars):
            text += next_char2(text, temperature)
        return text

    print(complete_text2("t"))
    return complete_text2, next_char2


if __name__ == "__main__":
    app.run()
