import marimo

__generated_with = "0.8.3"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo

    mo.md('''

    # Notebook setup

    ''')
    return mo,


@app.cell
def __(mo):
    from math import exp

    def dot(vector1, vector2):
        return sum(x * y for x, y in zip(vector1, vector2))

    def step(x):
        return 1 if x >= 0 else 0

    def sigmoid(x):
        return 1 / (1 + exp(-x))

    def tanh(x):
        return (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    mo.md('''

    Implementing some mathematical functions that we are going to use on this notebook:

    - Dot product
    - Step
    - Sigmoid (or logistic)
    - Hyperbolyc tangent (or tahn)

    ''')
    return dot, exp, sigmoid, step, tanh


@app.cell
def __(mo, sigmoid, step):
    import numpy as np
    import matplotlib.pyplot as plt

    def get_x_values():
        return [i * 0.1 for i in range(-100, 101)]  # From -10 to 10 with steps of 0.1

    def setup_plot():
        plt.figure(figsize=(6, 4)) # 6x4 inches
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)

    def show_plot(title, display_legend=False):
        if display_legend:
            plt.legend()

        plt.title(title)
        plt.show()

    def plot_xor():
        plt.figure(figsize=(5, 5))
        plt.xlabel('a')
        plt.ylabel('b')
        plt.grid(True)

        plt.scatter((0, 1), (0, 1), color='red', label='a XOR b = 0')
        plt.scatter((0, 1), (1, 0), color='blue', label='a XOR b = 1')

        show_plot('XOR', True)

    def plot_step():
        setup_plot()

        x_values = get_x_values()
        y_step = [step(x) for x in x_values]
        plt.plot(x_values, y_step, color='blue')

        show_plot('Step')

    def plot_sigmoid():
        setup_plot()

        x_values = get_x_values()
        y_values = [sigmoid(x) for x in x_values]   
        plt.plot(x_values, y_values, color='blue')

        show_plot('Sigmoid')

    def plot_sigmoid_bias():
        setup_plot()

        x_values = get_x_values()
        y_values = [sigmoid(x) for x in x_values]
        y_values_shifted = [sigmoid(x - 1) for x in x_values]

        plt.plot(x_values, y_values, label='sigmoid', color='blue')
        plt.plot(x_values, y_values_shifted, label='sigmoid (-1 bias)', color='red', linestyle='--')

        show_plot('Sigmoid', True)

    def plot_sigmoid_step():
        setup_plot()

        x_values = get_x_values()
        y_values_sigmoid = [sigmoid(x) for x in x_values]
        y_values_step = [step(x) for x in x_values]

        plt.plot(x_values, y_values_sigmoid, label='sigmoid', color='blue')
        plt.plot(x_values, y_values_step, label='step', color='red', linestyle='--')

        show_plot('Sigmoid and step', True)

    mo.md('''

    Definining some plot functions using matplot lib that we are going to use on this notebook.

    ''')
    return (
        get_x_values,
        np,
        plot_sigmoid,
        plot_sigmoid_bias,
        plot_sigmoid_step,
        plot_step,
        plot_xor,
        plt,
        setup_plot,
        show_plot,
    )


@app.cell
def __(mo):
    mo.md("""# Perceptron""")
    return


@app.cell
def __(mo):
    mo.mermaid('''

    graph TD
        x1["x₁"] --> w1["w₁"]
        x2["x₂"] --> w2["w₂"]
        x3["x₃"] --> w3["w₃"]
        b --> sum["∑"]
        w1 --> sum["∑"]
        w2 --> sum["∑"]
        w3 --> sum["∑"]
        sum["∑"] --> step
        step --> y

    ''')
    return


@app.cell
def __(dot, mo, step):
    def compute_perceptron(weights, bias, input):
        return step(dot(weights, input) + bias)

    mo.md('''

    The perceptron works like this:

    - Takes 𝑛 inputs of the same size, which can be scalar values or vectors (x₁, x₂, x₃)
    - Multiply each input by its own weight, a floating point number (w₁, w₂, w₃)
    - Sum the weighted inputs ( ∑ )
    Add a bias term to shift the function, sliding its line to either the left or right on a graph (b)
    - Pass the sum through the step function to get the final output (step)

    The output is calculated as:

    ```
    output = step(dot(x_values, w_values) + b)
    ```

    Let's understand each individual component of this computation below.

    ''')
    return compute_perceptron,


@app.cell
def __(mo):
    mo.md(
        r"""
        # Dot product

        For the weighted summation (simbolized by ∑ on the diagram), we should do:

        ```
        (x₁ * w₁) + (x₂ * w₂) + (x₃ * w₃)
        ```

        Let's take the following values as an example:

        Variable | Value
        -------- | -----
        x₁       | 1
        x₂       | 2
        x₃       | 3
        w₁       | 0.1
        w₂       | 0.2
        w₃       | 0.3

        This results in:

        ```
        (1 * 0.1) + (2 * 0.2) + (3 * 0.3) =
        0.1 + 0.4 + 0.9
        1.4
        ```

        Since x and w are of equal length, we can use the mathematical dot product operation to perform the summation we just saw above. Dot product takes two equal-length sequences of numbers and returns a single number:

        $$
        \mathbf{x} \cdot \mathbf{w} = 
        \begin{bmatrix}1 & 2 & 3\end{bmatrix}
        \cdot
        \begin{bmatrix}0.1 & 0.2 & 0.3\end{bmatrix}
        = 1.4
        $$
        """
    )
    return


@app.cell
def __(mo, plot_sigmoid_bias):
    plot_sigmoid_bias()

    mo.md('''

    # Bias

    A common technique to shift a function curve left or right is to add a (constant) value to its input, known as a "bias".

    To illustrate, consider the sigmoid function (in blue). Normally, for for x = 0, y = 1. By adding a bias of -1, we shift the function to the right, now for x = 0, y = 0.27 (which is just sigmoid(-1)).

    Don't worry about the sigmoid function for now, it will be introduced later on this notebook.

    ''')
    return


@app.cell
def __(mo, plot_step):
    plot_step()

    mo.md('''

    # Step function

    After computing the weighted sum, we pass the values to the step function:

    - for x < 0, y = 0
    - for x >= 0, y = 1

    It essentially works as an on/off switch, off when negative and on when positive.

    ''')
    return


@app.cell
def __(compute_perceptron, mo):
    def compute_not(value):
        not_weights = [-2.]
        not_bias = 1.
        return compute_perceptron(not_weights, not_bias, [value])

    print(f'NOT 0: {compute_not(0)}')
    print(f'NOT 1: {compute_not(1)}')

    mo.md('''

    # NOT gate

    a | NOT a
    - | -----
    0 | 1
    1 | 0

    The NOT gate that inverts the value.

    Using the perceptron algorithm we discussed earlier (weighted sum, bias, and step function), we can implement the NOT gate with a single weight and bias.

    ''')
    return compute_not,


@app.cell
def __(compute_perceptron, mo):
    def compute_and(value_a, value_b):
        and_weights = [2., 2.]
        and_bias = -3.
        return compute_perceptron(and_weights, and_bias, [value_a, value_b])

    def compute_or(value_a, value_b):
        or_weights = [2., 2.]
        or_bias = -1.
        return compute_perceptron(or_weights, or_bias, [value_a, value_b])

    print(f'0 AND 0: {compute_and(0, 0)}')
    print(f'0 AND 1: {compute_and(0, 1)}')
    print(f'1 AND 0: {compute_and(1, 0)}')
    print(f'1 AND 1: {compute_and(1, 1)}')
    print()
    print(f'0 OR 0: {compute_or(0, 0)}')
    print(f'0 OR 1: {compute_or(0, 1)}')
    print(f'1 OR 0: {compute_or(1, 0)}')
    print(f'1 OR 1: {compute_or(1, 1)}')

    mo.md('''

    # AND and OR gates

    a | b | a AND b | a OR b
    - | - | ------- | ------
    0 | 0 | 0       | 0
    0 | 1 | 0       | 1
    1 | 0 | 0       | 1
    1 | 1 | 1       | 1

    Using a couple of weights and a bias, we can compute AND and OR gates.

    ''')
    return compute_and, compute_or


@app.cell
def __(mo, plot_xor):
    plot_xor()

    mo.md('''

    # XOR gate

    a | b | a XOR b
    - | - | -------
    0 | 0 | 0
    0 | 1 | 1
    1 | 0 | 1
    1 | 1 | 0

    Unlike the other logic gates, the XOR gate is not linearly separable, that is, we cannot draw a single line to separate the true values from the false. As a result, a single perceptron cannot implement the XOR gate, a challenge known as "the XOR problem".

    Notice how the blue and red dots on the graph below cannot be separated by a single line.

    ''')
    return


@app.cell
def __(mo):
    mo.md(
        """
        # Neural network

        A single perceptron cannot compute the XOR gate because it is a non-linear function. However, by using more than one neuron and stacking them in layers, we can create what is called an "artificial neural network".

        Below is the structure of the neural network that we will implement in this notebook.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.mermaid('''

    graph TB;
        subgraph Input Layer
            x1[Input X₁];
            x2[Input X₂];
        end

        subgraph Hidden Layer
            h1[Hidden H₁];
            h2[Hidden H₂];
        end

        subgraph Output Layer
            y[Output Y];
        end

        x1 --> |Weight X₁H₁| h1;
        x1 --> |Weight X₁H₂| h2;

        x2 --> |Weight X₂H₁| h1;
        x2 --> |Weight X₂H₂| h2;

        h1 --> |Weight H₁Y| y;
        h2 --> |Weight H₂Y| y;

    ''')
    return


@app.cell
def __(mo):
    mo.md(
        """
        A typical neural network consists of an input layer, followed by one or more hidden layers, and an output layer.

        In the neural network we are building to compute the XOR logic gate, the input layer has two neurons corresponding to the two inputs of the XOR gate, and the output layer has one neuron, reflecting the single output of the gate. The hidden layer contains two neurons, which is the minimum number required to solve the XOR problem.

        "Neural networks can have hidden layers of varying sizes. When a network has multiple hidden layers, it is referred to as a "deep" network. In this "feed-forward" network, the input propagates in a single direction, from the input layer to the output layer. Additionally, because each neuron in one layer is connected to every neuron in the subsequent layer, the network is termed "fully connected" or "dense"."

        Each neuron typically includes an adjustable bias term (added to the summation of inputs), which is essential for shifting the activation function. However, for simplicity, biases are omitted from this diagram.
        """
    )
    return


@app.cell
def __(mo, plot_sigmoid_step):
    plot_sigmoid_step()

    mo.md('''

    # Sigmoid function

    Before tackling the XOR problem, we are introducing a key change: instead of using the step function after calculating the weighted sum of inputs and biases, we will use the sigmoid function.

    Observe in the plot below that both the sigmoid and step functions transition from 0 to 1. However, the key difference is that the sigmoid function provides a smooth, gradual transition, unlike the abrupt change seen in the step function.

    The sigmoid function is an example of an "activation function", which determines whether or not a neuron is activated based on its output.

    ''')
    return


@app.cell
def __(dot, mo, sigmoid):
    def compute_neuron(weights, inputs):
        bias = [1] # for this example we'll use a constant bias of 1
        input_with_bias = inputs + bias # making a list with inputs and bias
        return sigmoid(dot(weights, input_with_bias))

    def compute_network(layers, input):
        outputs = []

        for layer in layers:
            output = [compute_neuron(neuron_weights, input) for neuron_weights in layer]
            outputs.append(output)
            input = output

        return outputs

    def compute_xor(value_a, value_b):
        layers = [

            # hidden layer
            [[20., 20., -30], [20., 20., -10.]],

            # output layer
            [[-60., 60., -30.]]

        ]

        outputs = compute_network(layers, [value_a, value_b])
        last_output = outputs[-1][0]

        return round(last_output)

    print(f'0 XOR 0: {compute_xor(0, 0)}')
    print(f'0 XOR 1: {compute_xor(0, 1)}')
    print(f'1 XOR 0: {compute_xor(1, 0)}')
    print(f'1 XOR 1: {compute_xor(1, 1)}')

    mo.md('''

    # Solving the XOR problem

    Lastly, let's build our neural network and manually set the weights that we know in advance will allow the network to solve the XOR problem.

    ''')
    return compute_network, compute_neuron, compute_xor


if __name__ == "__main__":
    app.run()
