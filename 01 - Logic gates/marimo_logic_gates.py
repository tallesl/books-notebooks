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

    def sigmoid(x):
        return 1 / (1 + exp(-x))

    def tanh(x):
        return (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    def step(x):
        return 1 if x >= 0 else 0

    def dot(vector1, vector2):
        if len(vector1) != len(vector2):
            raise ValueError('Vectors must be of the same length')

        return sum(x * y for x, y in zip(vector1, vector2))

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

    def plot_sigmoid():
        setup_plot()

        x_values = get_x_values()
        y_values = [sigmoid(x) for x in x_values]
        plt.plot(x_values, y_values, color='blue')
        
        plt.title('Sigmoid')
        plt.show()

    def plot_sigmoid_bias():
        setup_plot()
        
        x_values = get_x_values()
        
        y_values = [sigmoid(x) for x in x_values]
        plt.plot(x_values, y_values, label='Sigmoid', color='blue')
        
        y_values_shifted = [sigmoid(x - 1) for x in x_values]
        plt.plot(x_values, y_values_shifted, label='Sigmoid (-1 bias)', color='red', linestyle='--')
        
        plt.title('Sigmoid')
        plt.legend()
        plt.show()

    def plot_step():
        setup_plot()
        
        x_values = get_x_values()
        y_step = [step(x) for x in x_values]
        plt.plot(x_values, y_step, color='blue')
        
        plt.title('Step')
        plt.show()

    mo.md('''

    Definining some plot functions using matplot lib that we are going to use on this notebook.

    ''')
    return (
        get_x_values,
        np,
        plot_sigmoid,
        plot_sigmoid_bias,
        plot_step,
        plt,
        setup_plot,
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
    def perceptron_output(weights, bias, input):
        return step(dot(weights, input) + bias)

    mo.md('''

    The perceptron works like this:

    - Takes 𝑛 same sized inputs, it can be scalar values or multidimensional matrices (x₁, x₂, x₃)
    - Multiply each input by its own weight, a floating point number (w₁, w₂, w₃)
    - Perform summation of the weighted inputs ( ∑ )
    - Add a bias to shift the function from its center (b)
    - Pass the value to the step function (step)

    The math looks like:

    ```
    output = step(dot(x_values, w_values) + b)
    ```

    Let's understand each part individually below.

    ''')
    return perceptron_output,


@app.cell
def __(mo):
    mo.md(
        r"""
        # Dot product

        For the weighted summation (simbolized by ∑ on the diagram), we should do:

        ```
        (x₁ * w₁) + (x₂ * w₂) + (x₃ * w₃)
        ```

        Let's take the following values as example:

        Variable | Value
        -------- | -----
        x₁       | 1
        x₂       | 2
        x₃       | 3
        w₁       | 0.1
        w₂       | 0.2
        w₃       | 0.3

        That gives us:

        ```
        (1 * 0.1) + (2 * 0.2) + (3 * 0.3) =
        0.1 + 0.4 + 0.9
        1.4
        ```

        Since x and w are equally sized, we can use the mathematical dot product function to perform the same summation we just saw above. Dot product is an operation that takes two equal-length sequences of numbers and returns a single number:

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

    A common technique to shift a function curve to the left or to the right is to add a (constant) value to its input. This added value is referred to as "bias".

    To illustrate, you can see the sigmoid function in blue, which for x = 0 we get y = 1. By subtracting 1 (bias = -1), we can shift the function to the right, now for x = 0 we get y = 0.27 (which is just sigmoid(-1)).

    Don't worry about the sigmoid function as of now, it will be introduced later on this notebook.

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
def __(mo):
    mo.md(
        """
        # Implementing a NOT gate

        Let's start with the NOT gate that simply inverts the value:

        a | NOT a
        - | -----
        0 | 1
        1 | 0

        Given the perceptron algorithm we saw on the previous section (weighted sum, bias, and step function), we can implement the same by using a single -2 weight.
        """
    )
    return


@app.cell
def __(perceptron_output):
    def compute_not(value):
        not_weights = [-2.]
        not_bias = 1.
        return perceptron_output(not_weights, not_bias, [value])

    print(f'NOT 0: {compute_not(0)}')
    print(f'NOT 1: {compute_not(1)}')
    return compute_not,


@app.cell
def __(mo):
    mo.md(
        """
        # Implementing AND and OR gates

        Here's a table of AND and OR:

        a | b | a AND b | a OR b
        - | - | ------- | ------
        0 | 0 | 0       | 0
        0 | 1 | 0       | 1
        1 | 0 | 0       | 1
        1 | 1 | 1       | 1

        And below we'll compute it with a couple of weights and the bias.
        """
    )
    return


@app.cell
def __(perceptron_output):
    def compute_and(value_a, value_b):
        and_weights = [2., 2.]
        and_bias = -3.
        return perceptron_output(and_weights, and_bias, [value_a, value_b])

    print(f'0 AND 0: {compute_and(0, 0)}')
    print(f'0 AND 1: {compute_and(0, 1)}')
    print(f'1 AND 0: {compute_and(1, 0)}')
    print(f'1 AND 1: {compute_and(1, 1)}')
    return compute_and,


@app.cell
def __(perceptron_output):
    def compute_or(value_a, value_b):
        or_weights = [2., 2.]
        or_bias = -1.
        return perceptron_output(or_weights, or_bias, [value_a, value_b])

    print(f'0 OR 0: {compute_or(0, 0)}')
    print(f'0 OR 1: {compute_or(0, 1)}')
    print(f'1 OR 0: {compute_or(1, 0)}')
    print(f'1 OR 1: {compute_or(1, 1)}')
    return compute_or,


@app.cell
def __(plt, sigmoid, step):
    def plot2():
    # Generate data points manually
        x_values = [i * 0.1 for i in range(-100, 101)]  # From -10 to 10 with step of 0.1
        y_sigmoid = [sigmoid(x) for x in x_values]
        y_step = [step(x) for x in x_values]

        # Plot the step function and sigmoid function
        plt.figure(figsize=(8, 6))
        plt.plot(x_values, y_sigmoid, label='Sigmoid Function', color='blue')
        plt.plot(x_values, y_step, label='Step Function', color='red', linestyle='--')
        plt.title('Step Function vs. Sigmoid Function')
        plt.xlabel('x')
        plt.ylabel('Output')
        plt.grid(True)
        plt.legend()
        plt.show()

    plot2()
    return plot2,


@app.cell
def __(mo):
    mo.md(r"""# The XOR problem""")
    return


@app.cell
def __(mo):
    mo.md("""# Neural network""")
    return


@app.cell
def __(mo):
    mo.md("""# Sigmoid function""")
    return


@app.cell
def __():
    return


@app.cell
def __(mo):
    mo.md(
        """
        # XOR gate


        a | b | a XOR b
        - | - | -------
        0 | 0 | 0
        0 | 1 | 1
        1 | 0 | 1
        1 | 1 | 0
        """
    )
    return


@app.cell
def __(perceptron_output):
    def compute_xor(value_a, value_b):
        
        xor_weights1 = [2., 2.]
        xor_bias1 = -1.
        
        xor_weights2 = [2., 2.]
        xor_bias2 = -1.
        
        xor_weights3 = [2., 2.]
        xor_bias3 = -1.
        
        output1 = perceptron_output(xor_weights1, xor_bias1, [value_a, value_b])
        output2 = perceptron_output(xor_weights2, xor_bias2, [output1])
        output3 = perceptron_output(xor_weights3, xor_bias3, [output2])

        return output3

    print(f'0 XOR 0: {compute_xor(0, 0)}')
    print(f'0 XOR 1: {compute_xor(0, 1)}')
    print(f'1 XOR 0: {compute_xor(1, 0)}')
    print(f'1 XOR 1: {compute_xor(1, 1)}')
    return compute_xor,


@app.cell
def __(mo):
    mo.md("""# Learning with backpropagation""")
    return


if __name__ == "__main__":
    app.run()
