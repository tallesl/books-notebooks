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


    mo.md(fr'''

    # 1 - Checking setup of needed libraries

    Importing the modules that we are going to use and verifying its versions:

    - Matplotlib version {matplotlib_version}
    - NumPy version: {numpy_version}
    - Pydot version: {pydot_version}
    - TensorFlow version: {tensorflow_version}
    - Keras version: {keras_version}

    On the next section we'll load the training and testing set of images and labels from the Fashion MNIST dataset.
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

    We can download Fashion MNIST with Keras itself:
    keras.datasets.fashion_mnist.load_data()`. This would download the dataset to
    here: `~/.keras/datasets/fashion-mnist.npz`.

    But let's load the file your own way instead.

    We are reproducing some load functions from [here](https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py),
    from same repository that hosts the dataset files (`.gz` files).

    ''')
    return gzip, load_images, load_labels


@app.cell
def __(load_images, load_labels, mo):
    training_labels = load_labels('fashion-mnist/train-labels-idx1-ubyte.gz')
    training_pixels = load_images('fashion-mnist/train-images-idx3-ubyte.gz')

    testing_labels = load_labels('fashion-mnist/t10k-labels-idx1-ubyte.gz')
    testing_pixels = load_images('fashion-mnist/t10k-images-idx3-ubyte.gz')

    label_description = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    mo.md(fr'''

    The Fashion MNIST dataset gives us 28x28 grayscale images. Each grayscale
    pixel is a single number going from 0 (white) to 255 (black). 28x28 gives us
    784 (consecutive) pixels.

    The number of labels (and images) should be equal the number of pixels divided
    by 784.

    Loaded training set:

    - Total training pixels: {len(training_pixels)}
    - Total training images: {len(training_pixels)/(28*28)}
    - Total training labels: {len(training_labels)}

    Loaded testing set:

    - Total testing pixels: {len(testing_pixels)}
    - Total testing images: {len(testing_pixels)/(28*28)}
    - Total testing labels: {len(testing_labels)}


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
        testing_labels,
        testing_pixels,
        training_labels,
        training_pixels,
    )


@app.cell
def __(label_description, mo, training_labels, training_pixels):
    import matplotlib.pyplot as plt

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
    print()
    plot_sample(last_image)
    print(label_description[last_label])

    mo.md(r'''

    Let's plot the first and last images of the training set as sample and check its label.

    We take 784 pixels from the training set pixels and then by using 
    NumPy's reshape method we change it to a 28x28 array.

    After that we handle our 2D array to Matplotlib and get the image plotted.

    ''')
    return (
        first_image,
        first_label,
        last_image,
        last_label,
        plot_sample,
        plt,
    )


@app.cell
def __(
    mo,
    testing_labels,
    testing_pixels,
    training_labels,
    training_pixels,
):
    total_training_images = int(len(training_pixels) / 784)
    training_images = training_pixels.reshape((total_training_images, 784))

    total_testing_images = int(len(testing_pixels) / 784)
    testing_images = testing_pixels.reshape((total_testing_images, 784))

    mo.md(fr'''

    Lastly, let's reshape our array of consecutive pixels into a 2x2 matrix (number of images x 784 pixels) with `.reshape(total images, 784)`

    That's the end of this section, on the next section we'll setup our Keras model that will train and predict based on our images and labels.

    - `training_images` shape: {training_images.shape}
    - `training_labels` shape: {training_labels.shape}
    - `testing_images` shape: {testing_images.shape}
    - `testing_labels` shape: {testing_labels.shape}

    ''')
    return (
        testing_images,
        total_testing_images,
        total_training_images,
        training_images,
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

        The input is the first layer of the network, which will receive the 784 pixel of each image from our dataset, without any transformation performed in our case.

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

        In a fully connected model such as ours, calculating the weights of a layer it's a simple matter of calculating previous layer units times this layer units. The number of biases is equal the number of units.

        From the input layer to the first hidden layer we have:

        - 235200 weights (784 units of the input layer x 300 of this layer)
        - 300 biases (same amount of units of this layer)

        From the first hidden layer to the second hidden layer we have:

        - 30000 weights (300 units of the input layer x 100 of this layer)
        - 100 biases (same amount of units of this layer)
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

        Lastly we have our output layer, which will output a number from 0 to 9, which is our model answer to the question "what is the category of the given image?".

        The output is from 0 to 9 because that is the range of categories of our dataset (Fashion MNIST). Our layer has 10 units, one for each category.

        Categorizing data is also referred as "classification".

        Calculating the weights and biases, from the last hidden layer to the output layer we have:

        - 1000 weights (100 units of the last hidden layer x 10 of this layer)
        - 10 biases (same amount of units of this layer)
        """
    )
    return


@app.cell
def __(mo):
    mo.md("""TODO explain softmax""")
    return


@app.cell
def __(keras, mo):
    layers= [
        keras.Input(shape=(784,), name='input'),
        keras.layers.Dense(units=300, activation='relu', name='Hidden #1'),
        keras.layers.Dense(units=100, activation='relu', name='Hidden #2'),
        keras.layers.Dense(units=10, activation='softmax', name='Output')
    ]

    model = keras.Sequential(layers, name='Fashion MNIST classification')

    mo.md(fr'''

    # 7 - Creating our model

    Let's instantiate our model now. After all the previous explanations, understanding the creation of this model should be straightforward. Note that "dense" just means the layer is fully connected.

    For further information check out the Keras documentation:

    - [keras.Sequential](https://keras.io/api/models/sequential/)
    - [keras.Input](https://keras.io/api/layers/core_layers/input/)
    - [keras.layers.Dense](https://keras.io/api/layers/core_layers/dense/)

    ''')
    return layers, model


@app.cell
def __(mo, model):
    model.summary()

    mo.md('''

    Keras provides a `.summary()` method that prints out a table with the model details. The "Param" count is the sum of weights and biases, the number should match the calculations that we did in the previous sections.

    More information on [keras.Model.summary](https://keras.io/api/models/model/#summary-method) documentation.

    ''')
    return


@app.cell
def __(keras, matplotlib, mo, model, plt):
    def plot_model_png():
        img = matplotlib.image.imread('model.png')
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_dtype=True, show_layer_names=True, show_layer_activations=True)

    plot_model_png()

    mo.md('''

    We also have a handy method available for exporting the model as a PNG image.


    More information on [keras.utils.plot_model](https://keras.io/api/utils/model_plotting_utils/) documentation.

    ''')
    return plot_model_png,


if __name__ == "__main__":
    app.run()
