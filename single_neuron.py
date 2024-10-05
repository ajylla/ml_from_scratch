import numpy as np
import matplotlib.pyplot as plt
from random import random


'''
OR, NOR, AND and NAND are able to be modelled
by this simple model.
'''

# [input, input, output]
data_OR = [[0, 0, 0],
           [0, 1, 1],
           [1, 0, 1],
           [1, 1, 1]]

data_NOR = [[0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 0]]

data_AND = [[0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 1]]

data_NAND = [[0, 0, 1],
             [0, 1, 1],
             [1, 0, 1],
             [1, 1, 0]]

'''
XOR and NXOR are not able to be modelled
by only a single neuron.
'''

data_XOR = [[0, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]]

data_NXOR = [[0, 0, 1],
             [0, 1, 0],
             [1, 0, 0],
             [1, 1, 1]]

'''
Change 'train' to any of the above to try them out!

'eps' is the small 'h' in the definition of a derivative,
i.e. how small you want the "wiggle" in a parameter to be.

'rate' is the "learning rate" i.e. in how big of steps
are the parameters updated in the negative derivative
direction.

Below you can also change the activation function
by setting 'activation' to be equal to 'sigmoid' or 'ReLU'.
'''

train = data_AND
eps = 1e-1
rate = 1e-1

'''
Model is a single "neuron" with two inputs (w1, w2), a bias (b) and an output.
GATE = w1*x1 + w2*x2 + b
'''

len_train = len(train)


def sigmoid(x):
    '''
    Sigmoid activation function
    '''
    return 1/(1 + np.exp(-x))


def ReLU(x):
    '''
    ReLU activation function
    '''
    return np.maximum(0, x)


# Here you set which one you want to use.
activation = sigmoid


def cost(w1, w2, b):
    '''
    The cost function, or the loss function. A measure of the
    goodness of the fit, nevertheless.
    I'm using a simple square distance average as the measure here.
    '''
    average = 0
    for i in train:
        x1, x2 = i[0], i[1]
        y = activation(w1*x1 + w2*x2 + b)
        d = (y - i[2])**2
        average += d

    return average/len_train


# Initialize all model parameters to random floats.
w1, w2, b = random(), random(), random()

# This holds the cost history for plotting purposes.
hist_cost = []
'''
This is the training loop.
The point is the differentiate the cost function,
and then nudge all the parameters in the decreasing
direction.
'''
for i in range(10000):
    c = cost(w1, w2, b)
    '''
    These are straight from the definition of a derivative.
    We nudge each parameter by eps amount, and figure out
    in which direction it increases.
    '''
    dw1 = (cost(w1 + eps, w2, b) - c)/eps
    dw2 = (cost(w1, w2 + eps, b) - c)/eps
    db = (cost(w1, w2, b + eps) - c)/eps

    '''
    And then go in the opposite (e.g. decreasing)
    direction, because we want to drive the cost to zero.
    '''
    w1 -= rate*dw1
    w2 -= rate*dw2
    b -= rate*db
    hist_cost = hist_cost + [cost(w1, w2, b)]

# Printing the results
print(f"Cost: {cost(w1, w2, b)}")
print("----------------")
print("Input1\tInput2\tResult")
for i in [0, 1]:
    for j in [0, 1]:
        print(f"{i}\t{j}\t{activation(i*w1 + j*w2 + b)}")

print("----------------")
print("With mathematical rounding:")
print("Input1\tInput2\tResult")
for i in [0, 1]:
    for j in [0, 1]:
        print(f"{i}\t{j}\t{int(activation(i*w1 + j*w2 + b)+0.5)}")

'''
Plotting part, to make the results a bit more visually interesting.
First plot is cost as a function of iteration.
Second plot is a 3D plot that shows the gate training data
as points, and also a 3D surface that represents the model
parameters, which is pretty cool.
'''
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(range(0, len(hist_cost)), hist_cost, c='k')
ax.set(xlabel="Iteration", ylabel="Cost")

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set(xlabel='Input 1', ylabel='Input 2', zlabel='Gate value',
       xticks=[0, 1], yticks=[0, 1], zticks=[0, 1])

for i in train:
    label = "Training data"
    ax.scatter(xs=i[0], ys=i[1], zs=i[2], c='k',
               label=label if label not in ax.get_legend_handles_labels()[1] else None)

x = np.arange(-0.2, 1.2, 0.01)
y = np.arange(-0.2, 1.2, 0.01)
x, y = np.meshgrid(x, y)
z = activation(x*w1 + y*w2 + b)
ax.plot_surface(x, y, z, alpha=0.25, label="Model")
ax.legend()
plt.show()
