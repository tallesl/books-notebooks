import marimo

__generated_with = "0.8.0"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __(mo):
    mo.md(r"""# 1 - Checking setup of needed libraries""")
    return


@app.cell
def __(mo):
    import matplotlib
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras

    mo.md(fr'''

    Importing the modules that we are going to use and verifying its versions:

    - Matplotlib version {matplotlib.__version__}
    - NumPy version: {np.version.version}
    - TensorFlow version: {tf.__version__}
    - Keras version: {keras.__version__}

    ''')
    return keras, matplotlib, np, tf


@app.cell
def __(mo):
    mo.md("""On the next section we'll load the training and testing set of images and labels from the Fashion MNIST dataset.""")
    return


@app.cell
def __(mo):
    mo.md("""# 2 - Loading the Fashion MNIST dataset""")
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        We can download Fashion MNIST with Keras itself: `keras.datasets.fashion_mnist.load_data()`.

        This would download the dataset to here: `~/.keras/datasets/fashion-mnist.npz`.

        But let's load the file your own way instead (see below).
        """
    )
    return


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

    ''')
    return testing_labels, testing_pixels, training_labels, training_pixels


@app.cell
def __(mo):
    label_description = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    mo.md(fr'''

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
    return label_description,


@app.cell
def __(label_description, mo, training_labels, training_pixels):
    import matplotlib.pyplot as plt

    def plot(image):
        plt.figure(figsize=(1,1)) # display the image as 1 inch x 1 inch
        plt.axis('off') # not plotting x and y axis with the image
        plt.imshow(image, cmap='gray')
        plt.show()

    image_sample = training_pixels[:784].reshape((28,28))
    label_sample = training_labels[0]

    plot(image_sample)
    print(label_description[label_sample])

    mo.md(r'''

    Let's plot the first image of the training set as sample and check its label.

    We take the first 784 pixels from the training set pixels and then by using 
    NumPy'sreshape method we change it to a 28x28 array.

    After that we handle our 2D array to Matplotlib and get the image plotted.

    ''')
    return image_sample, label_sample, plot, plt


@app.cell
def __(label_description, mo, plot, testing_labels, testing_pixels):
    another_image_sample = testing_pixels[-784:].reshape((28,28))
    another_label_sample = testing_labels[-1]

    plot(another_image_sample)
    print(label_description[another_label_sample])

    mo.md(r'''

    Another sample image, now the last from the training set.

    ''')
    return another_image_sample, another_label_sample


@app.cell
def __(mo, testing_pixels, training_pixels):
    total_training_images = int(len(training_pixels) / 784)
    training_images = training_pixels.reshape((total_training_images, 784))

    total_testing_images = int(len(testing_pixels) / 784)
    testing_images = testing_pixels.reshape((total_testing_images, 784))

    mo.md(r'''

    Lastly, let's reshape our array of consecutive pixels into a 2x2 matrix of number of images x 784 pixels with `.reshape(total images, 784)`

    ''')
    return (
        testing_images,
        total_testing_images,
        total_training_images,
        training_images,
    )


@app.cell
def __(
    mo,
    testing_images,
    testing_labels,
    training_images,
    training_labels,
):
    mo.md(fr'''

    That's the end of this section, on the next section we'll setup our Keras model that will train and predict based on our images and labels.

    - `training_images` shape: {training_images.shape}
    - `training_labels` shape: {training_labels.shape}
    - `testing_images` shape: {testing_images.shape}
    - `testing_labels` shape: {testing_labels.shape}

    ''')
    return


@app.cell
def __(mo):
    mo.md("""# 3 - Setting up our Keras model""")
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
        Above you can see a diagram with our proposed model comprising of an input layer, followed by two fully connected hidden layers, and lastly an output layer.

        Let's dedicate the sections below to understand the role of each layer of our model.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 3.1 - Input layer

        The input is the first layer of the network, which will receive the 784 pixel of each image from our dataset, without any transformation performed in our case.

        The input layer holds no weights or biases, it will simply pass the data forward to the next layer.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        ## 3.2 - Hidden layers

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
    mo.md(
        r"""
        ## 3.3 - Output layer

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
    mo.md("""## 3.4 - Activation functions""")
    return


@app.cell
def __(mo):
    mo.md(r"""## 3.5 - Creating our model""")
    return


@app.cell
def __(keras, mo):
    model = keras.Sequential([
        keras.Input(shape=(784,)),
        keras.layers.Dense(300, activation='relu'),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    mo.md(fr'''

    Let's instantiate our model now. After all the previous explanations, understanding the creation of this model should be straightforward. Note that "dense" just means the layer is fully connected.

    For further information check out the Keras documentation:

    - [keras.Sequential](https://keras.io/api/models/sequential/)
    - [keras.Input](https://keras.io/api/layers/core_layers/input/)
    - [keras.layers.Dense](https://keras.io/api/layers/core_layers/dense/)

    ''')
    return model,


if __name__ == "__main__":
    app.run()
