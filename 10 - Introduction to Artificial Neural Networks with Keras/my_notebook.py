import marimo

__generated_with = "0.8.0"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __(mo):
    import matplotlib
    import numpy as np
    import pydot
    import tensorflow as tf
    from tensorflow import keras

    matplotlib_version = matplotlib.__version__
    numpy_version = np.version.version
    pydot_version = pydot.__version__
    tensorflow_version = tf.__version__
    keras_version = keras.__version__


    mo.md(f'''

    # 1 - Checking setup of needed libraries

    Importing the modules that we are going to use and verifying its versions:

    - Matplotlib version {matplotlib_version}
    - NumPy version: {numpy_version}
    - Pydot version: {pydot_version}
    - TensorFlow version: {tensorflow_version}
    - Keras version: {keras_version}

    On the next section we'll load the training and validation set of images and labels from the Fashion MNIST dataset.

    ''')
    return (
        keras,
        keras_version,
        matplotlib,
        matplotlib_version,
        np,
        numpy_version,
        pydot,
        pydot_version,
        tensorflow_version,
        tf,
    )


@app.cell
def __(mo, np):
    import gzip

    def load_labels(path):
        with gzip.open(path, 'rb') as file:
            return np.frombuffer(file.read(), dtype=np.uint8, offset=8)

    def load_images(path):
        with gzip.open(path, 'rb') as file:
            return np.frombuffer(file.read(), dtype=np.uint8, offset=16)

    mo.md('''

    # 2 - Loading the Fashion MNIST dataset

    We can download Fashion MNIST with Keras itself with keras.datasets.fashion_mnist.load_data()`. This would download the dataset to `~/.keras/datasets/fashion-mnist.npz`.

    But let's load the file your own way instead.

    We are reproducing some load functions from [here](https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py), that's the same repository that hosts the dataset files (`.gz` files).

    ''')
    return gzip, load_images, load_labels


@app.cell
def __(load_images, load_labels, mo):
    training_labels = load_labels('fashion-mnist/train-labels-idx1-ubyte.gz')
    training_pixels = load_images('fashion-mnist/train-images-idx3-ubyte.gz')

    validation_labels = load_labels('fashion-mnist/t10k-labels-idx1-ubyte.gz')
    validation_pixels = load_images('fashion-mnist/t10k-images-idx3-ubyte.gz')

    label_description = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]

    mo.md(f'''

    The Fashion MNIST dataset gives us 28x28 grayscale images. Each grayscale pixel
    is a single number going from 0 (white) to 255 (black). 28x28 gives us 784
    (consecutive) pixels.

    The number of labels (and images) should be equal the number of (the entire 
    dataset pixels) divided by 784.

    Loaded training set:

    - Total training pixels: {len(training_pixels)}
    - Total training images: {len(training_pixels)/784}
    - Total training labels: {len(training_labels)}

    Loaded validation set:

    - Total validation pixels: {len(validation_pixels)}
    - Total validation images: {len(validation_pixels)/784}
    - Total validation labels: {len(validation_labels)}


    The labels goes from 0 to 9 and are categorized as follows:

    Label | Description
    ----- | -----------
    0     | {label_description[0]}
    1     | {label_description[1]}
    2     | {label_description[2]}
    3     | {label_description[3]}
    4     | {label_description[4]}
    5     | {label_description[5]}
    6     | {label_description[6]}
    7     | {label_description[7]}
    8     | {label_description[8]}
    9     | {label_description[9]}

    ''')
    return (
        label_description,
        training_labels,
        training_pixels,
        validation_labels,
        validation_pixels,
    )


@app.cell
def __(label_description, mo, training_labels, training_pixels):
    import matplotlib.pyplot as plt
    from time import sleep

    def plot_sample(image):
        plt.figure(figsize=(1,1)) # display the image as 1 inch x 1 inch
        plt.axis('off') # not plotting x and y axis with the image
        plt.imshow(image, cmap='gray')
        plt.show()

    first_image = training_pixels[:784].reshape((28,28))
    first_label = training_labels[0]

    last_image = training_pixels[-784:].reshape((28,28))
    last_label = training_labels[-1]

    plot_sample(first_image)
    print(label_description[first_label])

    sleep(0.01) # ugly fix to ensure plotting in the correct order
    print()

    plot_sample(last_image)
    print(label_description[last_label])

    mo.md('''

    Let's plot the first and last images of the training set as sample and check its label.

    We take 784 pixels from the training set pixels and then by using NumPy's reshape method we change it to a 28x28 array.

    After that we handle our 2D array to Matplotlib and get the image plotted.

    ''')
    return (
        first_image,
        first_label,
        last_image,
        last_label,
        plot_sample,
        plt,
        sleep,
    )


@app.cell
def __(
    mo,
    training_labels,
    training_pixels,
    validation_labels,
    validation_pixels,
):
    total_training_images = int(len(training_pixels) / 784)
    training_images = training_pixels.reshape((total_training_images, 784))

    total_validation_images = int(len(validation_pixels) / 784)
    validation_images = validation_pixels.reshape((total_validation_images, 784))

    mo.md(f'''

    Lastly, let's reshape our array of consecutive pixels into a 2x2 matrix (number of images x 784 pixels) with `.reshape(total images, 784)`

    That's the end of this section, on the next section we'll setup our Keras model that will train and predict based on our images and labels.

    - `training_images` shape: {training_images.shape}
    - `training_labels` shape: {training_labels.shape}
    - `validation_images` shape: {validation_images.shape}
    - `validation_labels` shape: {validation_labels.shape}

    ''')
    return (
        total_training_images,
        total_validation_images,
        training_images,
        validation_images,
    )


@app.cell
def __(mo):
    mo.md(
        """
        # 3 - Model architecture

        Below you can see a diagram with our proposed model comprising of an input layer, followed by two fully connected hidden layers, and lastly an output layer.
        """
    )
    return


@app.cell
def __(mo):
    mo.mermaid('''

    graph TD
        in["input layer<br>shape: (784,)"]
        in --> hidden1["fully connected hidden layer<br>units: 300<br>activation function: ReLU"]
        hidden1 --> hidden2["fully connected hidden layer<br>units: 100<br>activation function: ReLU"]
        hidden2 --> out["output layer<br>units: 10<br>activation function: Softmax"]

    ''')
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # 4 - Input layer

        The input is the first layer of the network, which will receive the 784 pixel of each image from our dataset.

        The input layer holds no weights or biases, it will simply pass the data forward to the next layer.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        # 5 - Hidden layers and ReLU

        Our model contains two hidden layers, with 300 units and 100 units respectively. "Units" are also sometimes referred as "neurons".

        In a fully connected model such as ours, calculating the weights of a layer it's a simple matter of calculating previous layer units times the current layer units. The number of biases is equal the number of units of the current layer.

        From the input layer to the first hidden layer we have:

        - 235200 weights (784 units of the input layer x 300 units of hidden layer #1)
        - 300 biases (same amount of units of hidden layer #1)

        From the first hidden layer to the second hidden layer we have:

        - 30000 weights (300 units of hidden layer #1 x 100 units of hidden layer #2)
        - 100 biases (same amount of units of hidden layer #2)
        """
    )
    return


@app.cell
def __(mo):
    mo.md("""TODO explain relu""")
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # 6 - Output layer and softmax

        Lastly we have our output layer, which outputs 10 probabilities, one for each of the 10 categories of the dataset. The number with the highest probability is the model answer to "what is the category of the given image?". The higher the number, the more sure the model is that it belongs to such category.

        Categorizing data is also referred as "classification".

        Calculating the weights and biases, from the last hidden layer to the output layer, we have:

        - 1000 weights (100 units of the hidden layer #2 x 10 units of the output layer)
        - 10 biases (same amount of units of the output layer)
        """
    )
    return


@app.cell
def __(mo):
    mo.md("""TODO explain softmax""")
    return


@app.cell
def __(mo):
    mo.md(
        """
        # 7 - Setting up a seed

        Unless you have special hardware, computers use pseudo-random number generators. These generators use mathematical algorithms to produce sequences of numbers that seem random but are actually predictable if you know the seed value. By setting an initial seed, you ensure that every time you start the generator with the same seed, you get the same sequence of numbers.

        In other words, setting a seed helps with analysis and debugging by ensuring reproducible results.
        """
    )
    return


@app.cell
def __(mo):
    seed_textbox = mo.ui.text(value='123', label='Seed value to be used (feel free to change it):')
    seed_textbox
    return seed_textbox,


@app.cell
def __(mo, np, seed_textbox, tf):
    import random

    seed = int(seed_textbox.value)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    mo.md(f'''

    We just set up {seed} as the seed for random values for Python, NumPy, and TensorFlow.

    ''')
    return random, seed


@app.cell
def __(keras, mo):
    layers= [
        keras.Input(shape=(784,), name='input'),
        keras.layers.Rescaling(scale=1./255, name='scale_to_0_1_range'),
        keras.layers.Dense(units=300, activation='relu', name='hidden_1'),
        keras.layers.Dense(units=100, activation='relu', name='hidden_2'),
        keras.layers.Dense(units=10, activation='softmax', name='output')
    ]

    model = keras.Sequential(layers, name='fashion_mnist')

    mo.md(fr'''

    # 7 - Creating our model

    Let's instantiate our model now. After all the previous explanations, understanding the creation of this model should be straightforward. Note that "dense" just means the layer is fully connected.

    Naming the model and the layers is optional, but when providing names make sure to not use spaces, else you'll get a "not a valid root scope name" error later.

    For further information check out the Keras documentation:

    - [keras.Sequential](https://keras.io/api/models/sequential/)
    - [keras.Input](https://keras.io/api/layers/core_layers/input/)
    - [keras.layers.Rescaling](https://keras.io/api/layers/preprocessing_layers/image_preprocessing/rescaling/)
    - [keras.layers.Dense](https://keras.io/api/layers/core_layers/dense/)

    ''')
    return layers, model


@app.cell
def __(mo):
    mo.md(
        """
        You might have noticed a "rescaling" layer between the input layer and the first hidden layer. It's actually not a layer per se, it doesn't have weights or biases, it's a transformation configured on the model that happens right after the input layer passes forwards the data.

        It multiplies the 0 to 255 value by (1/255), converting it to a floating pointer number from 0.0 to 1.0.

        Using floats allows for fine-grained representation of data, which is needed for effective learning and ensures that the model can capture subtle variations in the data.
        """
    )
    return


@app.cell
def __(mo):
    def valid_integer(value, min, max):
        if not value.isdecimal():
            print(f'"{value}" is invalid!')
            return

        int_value = int(value)

        if int_value < min or int_value > max:
            print(f'"{value}" is invalid!')
        else:
            return int_value

    def rescaling_textbox_change(value):
        int_value = valid_integer(value, 0, 255)

        if int_value:
            print(f'{int_value} x (1./255) = {int_value * (1./255)}')

    rescaling_textbox = mo.ui.text(label='Enter a value from 0 to 255:', on_change=rescaling_textbox_change)
    rescaling_textbox
    return rescaling_textbox, rescaling_textbox_change, valid_integer


@app.cell
def __(mo, model):
    model.summary()

    mo.md('''

    Keras provides a `.summary()` method that prints out a table with the model details. The parameter count represents the sum of weights and biases, the number should match the calculations that we did in the previous sections.

    More information on [keras.Model.summary](https://keras.io/api/models/model/#summary-method) documentation.

    ''')
    return


@app.cell
def __(keras, matplotlib, mo, model, plt):
    def plot_model_png():
        img = matplotlib.image.imread('model.png')
        plt.figure(figsize=(8, 8)) # display the image as 8 inch x 8 inch
        plt.axis('off') # not plotting x and y axis with the image
        plt.imshow(img)
        plt.show()

    keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_dtype=True, show_layer_names=True, show_layer_activations=True)

    plot_model_png()

    mo.md('''

    We also have a handy method available for exporting the model as a PNG image.

    More information on [keras.utils.plot_model](https://keras.io/api/utils/model_plotting_utils/) documentation.

    ''')
    return plot_model_png,


@app.cell
def __(mo, model):
    def compile_model():
        model.compile(
            optimizer='sgd',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    compile_model()

    mo.md('''

    # 8 - Compiling the model

    TODO explain what compiling the model means

    More information on [keras.Model.compile](https://keras.io/api/models/model_training_apis#compile-method) documentation.

    ''')
    return compile_model,


@app.cell
def __(mo):
    mo.md("""TODO explain optimizer and sgd""")
    return


@app.cell
def __(mo):
    mo.md("""TODO explain loss and sparse categorical cross entropy""")
    return


@app.cell
def __(mo):
    mo.md(r'''

    # 9 - Training the model

    TODO explain training

    ''')
    return


@app.cell
def __(
    mo,
    model,
    plt,
    training_images,
    training_labels,
    valid_integer,
    validation_images,
    validation_labels,
):
    def plot_training(history):  
        plt.figure(figsize=(8, 5))  # Define the figure here before plotting
        
        for key in history.keys():
            plt.plot(history[key], label=key)
        
        plt.grid(True)
        plt.ylim(0, 1)
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Learning Curve')
        plt.legend()
        plt.show()

    def train(value):
        epochs = valid_integer(value, 1, 1000)
        if epochs:
            history = model.fit(
                x=training_images,
                y=training_labels,
                validation_data=(validation_images, validation_labels),
                epochs=epochs
            )

            plot_training(history.history)

    training_form = mo.ui.text(label='Epochs to train:', value='10').form(on_change=train)
    training_form

    return plot_training, train, training_form


@app.cell
def __(mo):
    mo.md("""If you want to reset the model with random weights, to experiment on your own, you can use the button below.""")
    return


@app.cell
def __(compile_model, keras, mo, model, tf):
    def reset_weights(_):
        for layer in model.layers:
            if isinstance(layer, keras.layers.Dense):
                print(f'Resetting weights of {layer.name} layer.')
                kernel_shape = layer.kernel.shape
                bias_shape = layer.bias.shape

                # Create new initializers
                kernel_initializer = tf.keras.initializers.get(layer.kernel_initializer)
                bias_initializer = tf.keras.initializers.get(layer.bias_initializer)

                # Initialize new weights
                new_kernel = kernel_initializer(shape=kernel_shape)
                new_bias = bias_initializer(shape=bias_shape)

                # Set the weights
                layer.set_weights([new_kernel, new_bias])

        print('Recompiling model.')
        compile_model()
        print('Done.')

    button = mo.ui.run_button(on_change=reset_weights)
    button
    return button, reset_weights


@app.cell
def __(mo):
    mo.md(r"""# 10 - Classifying an image""")
    return


@app.cell
def __(
    label_description,
    matplotlib,
    mo,
    model,
    plt,
    valid_integer,
    validation_images,
    validation_labels,
):
    def plot_prediction(image, label, prediction):
        plt.figure(figsize=(6, 3))

        # subplot 1: label and image
        plt.subplot(1, 2, 1)
        plt.imshow(image.reshape(28, 28), cmap='gray')
        plt.title(f'{label} ({label_description[label]})')
        plt.axis('off') # not plotting x and y axis with the image

        # subplot 2: graph with prediction
        plt.subplot(1, 2, 2)
        plt.title('Prediction')
        plt.xlabel('Category')
        plt.ylabel('Probability')
        plt.bar(range(10), prediction, color='blue') # 0 to 10 bar chart
        plt.xticks(range(10)) # making sure X axis have 0 to 9 displayed
        plt.ylim(0, 1.0) # making sure Y axis goes up to the maximum limit (1.0)

        # formatting Y axis as percentage
        formatter = matplotlib.ticker.PercentFormatter(xmax=1.0)
        plt.gca().yaxis.set_major_formatter(formatter)

        plt.tight_layout() # automatically adjust spaces between subplots
        plt.show()

    def predict(i):
        int_value = valid_integer(i, 0, len(validation_images))

        image = validation_images[int_value]
        label = validation_labels[int_value]

        reshaped_image = image.reshape(1, 784)

        prediction = model.predict(reshaped_image, verbose=0) # setting zero to verbose suppress the progress bar

        plot_prediction(image, label, prediction[0])

    prediction_form = mo.ui.text(label='Index from the validation to perform prediction:', value='12').form(on_change=predict)
    prediction_form
    return plot_prediction, predict, prediction_form


@app.cell
def __(mo):
    mo.md(r"""# 11 - Saving the model""")
    return


@app.cell
def __(mo):
    mo.md(r"""# 12 - Visualizing the learning with TensorBoard""")
    return


@app.cell
def __(mo):
    mo.md(r"""# 13 - Converting it to a TensorFlow Lite model and using it""")
    return


if __name__ == "__main__":
    app.run()
