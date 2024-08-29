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

    After doing our weighted sum, we'll pass the values to the step function:

    - for x < 0, y = 0
    - for x >= 0, y = 1

    It basically works as a on/off switch, off when negative and on when positive.

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

    The NOT gate that simply inverts the value.

    Given the perceptron algorithm we saw on the previous section (weighted sum, bias, and step function), we can implement the gate by using a single weight (and bias).

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

    With a couple of weights (and one bias), we can compute AND and OR gates.

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

    Unlike the other logic gates, the XOR gate is not linearly separable, that is, we cannot draw a single line to separate the true values from the false. Due to that, we cannot use a single perceptron to implement it. This is known as "the XOR problem".

    See how you cannot separate the blue and red dots on the graph below with a single line.

    ''')
    return


@app.cell
def __(mo):
    mo.md(
        """
        # Neural network

        While we cannot use a single perceptron to compute the XOR gate, we can compute it by using more than one neuron. Stacking up layers of artificial neurons and connecting them up makes what is called an "artificial neuron network".

        Below we can see the network that we'll implement on this notebook.
        """
    )
    return


@app.cell
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
        A typical neural network consists of an input layer (with 𝑛 neurons), followed by 𝑛 hidden layers (with 𝑛 neurons each), followed by an output layer (with 𝑛 neurons).

        On our network that we are building, to compute the XOR logic gate, we have two neurons at the input layer because the logic gate also has two inputs, and a single output just as the gate. The two neurons at the hidden layer is somewhat arbitrary, in this case 2 because is the least amount that we need to solve this problem.

        Neural networks can have many different sized hidden layers. When having many, it's said that is a "deep" network. This kind of network, in which the input propagates from the input layer in a single direction to the output layer is called "feed forward". And lastly, since we have all neurons from one layer to connect to all neurons of the next layer, it's said to be "fully connected" or "dense".

        Each neuron contains it's own adjustable bias (that is added to the input summation as we saw previously), but it was omitted from the diagram for simplicity.
        """
    )
    return


@app.cell
def __(mo, plot_sigmoid_step):
    plot_sigmoid_step()

    mo.md('''

    # Sigmoid function


    Before tackling the XOR, there is just extra piece that we are changing: instead of using the step function after the input and bias summation is calculated, we are going to use the sigmoid function this time.

    Note on the plot below that the sigmoid is similar to the step function, both going from 0 to 1. The main difference is that the sigmoid is *smooth*, it doesn't have the sharp change from 0 to 1 that we see on the step function.

    This function that is used to compute the final output of a neuron is called an "activation function", because it basically dictates if the neuron is being activated or not.

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

    Now let's create our network and put some weights that we know beforehand that is able to make XOR work.

    ''')
    return compute_network, compute_neuron, compute_xor


if __name__ == "__main__":
    app.run()
