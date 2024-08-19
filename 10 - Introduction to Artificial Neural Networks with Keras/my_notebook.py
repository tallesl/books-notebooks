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

    We can download Fashion MNIST with Keras itself with `keras.datasets.fashion_mnist.load_data()`. This would download the dataset to `~/.keras/datasets/fashion-mnist.npz`.

    But let's load the file our own way instead.

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
    dataset) pixels divided by 784.

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

    We are taking 784 pixels from the training set and then by using NumPy's reshape method we change it to a 28x28 array.

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
        in --> hidden1["fully connected hidden layer #1<br>units: 300<br>activation function: ReLU"]
        hidden1 --> hidden2["fully connected hidden layer #2<br>units: 100<br>activation function: ReLU"]
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
def __(mo, np, plt):
    def plot_relu():
        x_values = np.linspace(-10, 10, 100)
        y_values = relu(x_values)

        plt.figure(figsize=(6, 4))
        plt.plot(x_values, y_values, label='ReLU(x)')
        plt.title('Rectified Linear Unit (ReLU)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.show()

    def relu(x):
        return np.maximum(0, x)

    plot_relu()

    mo.md('''

    The Rectified Linear Unit (ReLU) is defined as:

    ```
    ReLU(x) = max(0, x)
    ```

    It just means:

    - If `x` is positive, returns x.
    - If `x` is negative or zero, returns zero.

    It's simplicity makes it computationally efficient and it's one of the most used activation functions for hidden layers.

    ''')
    return plot_relu, relu


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
    mo.md(
        r"""
        The Softmax function is used to convert a set of raw scores (also called "logits") into probabilities. It's commonly used in the final layer of a neural network for classification tasks where the model must pick an option  between multiple classes.

        Its calculation can be broken down into three steps. Let's walk through the probability calculation given the following values:

        Category   | Value
        --------   | -----
        Category A | 2.0
        Category B | 1.0
        Category C | 0.1

        **Step 1: Calculate the exponential**

        - exp(2.0) = 7.389
        - exp(1.0) = 2.718
        - exp(0.1) = 1.105

        **Step 2: Sum the exponentials**

        7.389 + 2.718+ 1.105 = 11.212

        **Step 3: Calculate the proportion of the sum**

        - exp(2.0) / sum = 7.389 / 11.212 = 0.659
        - exp(1.0) / sum = 2.718 / 11.212 = 0.242
        - exp(0.1) / sum = 1.105 / 11.212 = 0.099

        Here are the final probabilities that softmax gave us:

        Category   | Value | Probability
        --------   | ----- | -----------
        Category A | 2.0   | 0.659 (or 66%)
        Category B | 1.0   | 0.242 (or 24%)
        Category C | 0.1   | 0.099 (or 10%)
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        # 7 - Setting up a seed

        Unless you have special hardware, computers use pseudo-random number generators. These generators use mathematical algorithms to produce sequences of numbers that seem random but are actually predictable if you know the seed value.

        By setting an initial seed, you ensure that every time you start the generator with the same seed, you get the same sequence of numbers. Setting a seed helps with analysis and debugging by ensuring reproducible results.
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
def __(mo, tf):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)

    mo.md('''

    # 8 - Creating our model

    First, before creating our model, let's ensure that TensorFlow does not allocate all of our GPU memory by calling `.set_memory_growth(gpu, True)`.

    More information on [tf.config.experimental.set_memory_growth](https://www.tensorflow.org/api_docs/python/tf/config/experimental/set_memory_growth) documentation.

    ''')
    return gpu, gpus


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

    Let's instantiate our model now. After all the previous explanations, understanding our model architecture should be straightforward. Note that "dense" just means the layer is fully connected.

    Naming the model and the layers is optional, but when providing names make sure to not use spaces, else you'll get a "not a valid root scope name" error later.

    More information on Keras documentation:

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
        If you check the previous code, you'll notice a "rescaling" layer between the input layer and the first hidden layer. It's actually not a layer per se, it doesn't have weights or biases, it's a transformation configured on the model that happens right after the input layer passes forwards the data.

        It multiplies the 0 to 255 value by 1/255, converting it to a floating pointer number from 0.0 to 1.0 (which can also be seem as a 0% to 100% range).

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

    # 9 - Compiling the model, SGD, and cross-entropy

    Compiling a model means setting up all the need settings for its training:

    - The optimization algorithm, stochastic gradient descent (SGD) in our case.
    - The loss function, sparse categorical crossentropy in our case.
    - The metrics displayed during training, just accuracy in our case (correct predictions / total predictions).

    More information on [keras.Model.compile](https://keras.io/api/models/model_training_apis#compile-method) documentation.

    ''')
    return compile_model,


@app.cell
def __(mo, np, plt):
    def plot_gradient_descent():
        def compute_y(x):
            return x**2 + 4*x + 4

        def compute_gradient(x):
            return 2*x + 4

        def perform_gradient_descent(learning_rate, initial_x, num_iterations):
            x = initial_x
            x_history = [x]
            for _ in range(num_iterations):
                gradient = compute_gradient(x)
                x = x - learning_rate * gradient
                x_history.append(x)
            return x_history

        learning_rate = 0.1
        initial_x = 5
        num_iterations = 20

        x_history = perform_gradient_descent(learning_rate, initial_x, num_iterations)
        x_values = np.linspace(-6, 6, 100)

        y_history = [compute_y(x) for x in x_history]
        y_values = compute_y(x_values)

        plt.plot(x_values, y_values, label='y = x^2 + 4x + 4')
        plt.scatter(x_history, y_history, color='red', label='steps')

        plt.title('Gradient Descent Example')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.show()

    plot_gradient_descent()

    mo.md('''

    Optimization algorithms are used during the training phase to minimize the loss function. For each iteration (or step), the loss is minimized by adjusting the model's parameters in the direction that reduces the loss.

    Gradient descent is the most commonly used optimization algorithm. It calculates the direction of the steepest decrease in the loss function, which is the goal of training: to reduce the loss.

    The calculated gradient is then multiplied by the learning rate, which determines how much the weights will be adjusted (increased or decreased) in each step.

    ''')
    return plot_gradient_descent,


@app.cell
def __(mo):
    mo.md(
        """
        Stochastic Gradient Descent (SGD) is a variant of gradient descent that uses batches of randomly selected items (hence stochastic) from the dataset to compute gradients.

        If you have a dataset with 60,000 items (like Fashion MNIST) and use a batch size of 32 (Keras default), Keras will split the dataset into 1875 batches. The model will compute the forward pass, gradient, and loss 1875 times. 1875 weight updates will be performed as well, per epoch.

        The use of batches reduces the memory and computation required in a single step during training. It's especially useful when using large datasets which may require too much computation or can be too big to fit in memory.

        Due to its stochastic nature, it may not always converge to the global minimum loss. Also, since updates are more frequent, a smaller learning rate is recommended.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        Loss, also known as error or cost, measures how far the predictions are from the actual values. This value is used not only to assess model performance but also to guide the learning process, as it is backpropagated from the output layer to adjust the weights in previous layers.

        Loss functions are chosen based on the type of problem the model is solving, such as regression or classification. We'll use the sparse crossentropy function while training our model.

        Its calculation is simple: -log(x), where x is the predicted probability for the expected correct category. Probabilities for the incorrect categories are ignored in the loss calculation.

        Example:

        Predicted  for A | Predicted for B | Predicted for C | Correct category | Loss
        ---------------- | --------------- | --------------- | ---------------- | ----
        0.4              | 0.1             | 0.5             | A                | -log(0.4) = 0.398
        0.2              | 0.7             | 0.1             | B                | -log(0.7) = 0.155
        0.3              | 0.4             | 0.3             | C                | -log(0.3) = 0.523
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # 10 - Training the model

        While training, TensorFlow displays a handy progress bar for the current epoch being training, along with the values:

        - `accuracy`: the accuracy on the training set (0.45 means 45% accuracy for example).
        - `loss`: the calculated loss on the training set, calculated by our loss function (the loss should decrease over time if the learning is effective)
        - `val_accuracy`: the accuracy on the validation set
        - `val_loss`: the calculated loss on the validation set

        We also plot the variation of those values over time after the training is finished (all epochs are finish).

        The Fashion MNIST dataset already gives a set for training and another for validation separately. But when we have the desired training and validation data loaded in a single dataset, we can use the parameter `validation_split` to let Keras split the dataset for us. For instance, a `validation_split=0.15` reserves the last 15% portion of the dataset to be used as validation.

        More information on [keras.Model.fit](https://keras.io/api/models/model_training_apis/#fit-method) documentation.
        """
    )
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
        epochs = valid_integer(value, 1, 999999)
        if epochs:
            history = model.fit(
                x=training_images,
                y=training_labels,
                validation_data=(validation_images, validation_labels),
                epochs=epochs,
            )

            plot_training(history.history)

    training_form = mo.ui.text(label='Epochs to train:', value='30').form(submit_button_label='Start training', on_change=train)
    training_form
    return plot_training, train, training_form


@app.cell
def __(mo):
    mo.md(
        r"""
        When creating and training a model, it's important to pay attention to its hyperparameters. These are, roughly speaking, the model's settings. The term "hyper" is used to distinguish them from "parameters," which some people refer to as the model's internal values (like weights and biases).

        If your model is not achieving the expected results, it's worth experimenting with different hyperparameters to see if performance improves.

        Here are the hyperparameters of our model:

        Hyperparameter         | Value
        --------------         |------
        Activation function    | ReLU (hidden layers), Softmax (output layer)
        Loss function          | Sparse categorical crossentropy
        Optimization algorithm | Stochastic Gradient Descent (SGD)
        Data type              | float32 (Keras default)
        Learning rate          | 0.01 (Keras default)
        Batch size             | 32 (Keras default)
        Training epochs        | 30

        These settings are well-suited for our proposed classification problem, but you are welcome to experiment with different values and observe how the model behaves.

        For your convenience, there's a button below to reset the model to its initial state with random weights.

        """
    )
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

    button = mo.ui.run_button(label='Reset Model Weights', on_change=reset_weights)
    button
    return button, reset_weights


@app.cell
def __(mo):
    mo.md(
        r"""
        # 11 - Classifying an image

        Now that the training is done, let's make our model predict what the category for a single given image.

        Below enter an index (0 to 9999) of an image to be picked from the validation set and hit Submit to see the results.

        On the left side you'll see the chosen image and its category, on the right side there's probabilities calculated by the model for each category.

        More information on [keras.Model.predict](https://keras.io/api/models/model_training_apis/#predict-method) documentation.
        """
    )
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

    prediction_form = mo.ui.text(label='Index of the image set to be picked from the validation set:', value='12').form(submit_button_label='Predict', on_change=predict)
    prediction_form
    return plot_prediction, predict, prediction_form


@app.cell
def __(mo):
    mo.md(
        """
        # 12 - Earling stopping

        Keras allows us to configure the model to stop the training earlier, that is, to stop it when the training is good enough even if we didn't reach the number of epochs we set.

        We can do that by setting an `EarlyStopping` callback. It also contains a parameter suggestively named "patience": how many further attempts it will try until giving up and stopping.

        We just setup the early stop callback on the model (with 5 for patience) and you can now try it out for yourself:

        - Click the button below to reset the model.
        - Go back to the training cell and set epoch to a high number (like 999).
        - Start the training and see when it will stop.

        More information on [keras.callbacks.EarlyStopping](https://keras.io/api/callbacks/early_stopping/) documentation.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # 13 - Saving and loading the model

        Saving and loading is straightforward, it's just a matter of calling `model.save('my_model.keras')` and `model.load('my_model.keras')`.

        More information on [keras.Model.save](https://keras.io/api/models/model_saving_apis/model_saving_and_loading/#save-method) and [keras.Model.load](https://keras.io/api/models/model_saving_apis/model_saving_and_loading/#loadmodel-function) documentation.
        """
    )
    return


@app.cell
def __(mo, model):
    def save_model(filename):
        model.save(filename)
        print(f'Saved model as "{filename}".')

    save_model_form = mo.ui.text(label='Filename: ', value='fashion_mnist.keras').form(submit_button_label='Save model', on_change=save_model)
    save_model_form
    return save_model, save_model_form


@app.cell
def __(mo):
    mo.md(
        """
        Keras provides a `ModelCheckpoint` callback that you can pass to your model to save it after every epoch. To ensure that only the best version of your model is saved, set the `save_best_only` parameter to `True`.

        Even better, you can combine this with the `EarlyStopping` callback we discussed earlier. This way, you can leave your computer running, knowing that the training will automatically stop when it's no longer improving and that the best model will be saved when the training is complete!

        More information on [keras.callbacks.ModelCheckpoint](https://keras.io/api/callbacks/model_checkpoint/) documentation.

        """
    )
    return


if __name__ == "__main__":
    app.run()
