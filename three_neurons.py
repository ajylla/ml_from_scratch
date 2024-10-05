import numpy as np
import matplotlib.pyplot as plt
from copy import copy

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

data_XOR = [[0, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]]

data_NXOR = [[0, 0, 1],
             [0, 1, 0],
             [1, 0, 0],
             [1, 1, 1]]


train = data_XOR
eps = 1e-2
rate = 1e-1

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


activation = sigmoid


def forward(input, w1, b1, w2, b2):
    '''
    This time the maths is a bit more complicated so I wrapped
    it in this function.
    Dimensions of matrices:
        w1 (2x2), b1 (1x2), w2 (1x2), b2 (1x1)
    '''
    # 'z' here is all of the matrix maths after
    # the first two neurons, or the first "layer".
    z = activation(np.matmul(w1, input) + b1)
    # 'y' is what the last neuron, or the last layer,
    # outputs.
    y = activation(np.matmul(w2, z) + b2)
    return y

def cost(w1, b1, w2, b2):
    '''
    The cost function is essentially the same
    as last time.
    '''
    average = 0
    for i in train:
        input = np.array([i[0], i[1]])
        y = forward(input, w1, b1, w2, b2)
        d = (y - i[2])**2
        average += d
    return average/len_train

# Initialise all parameters.
w1 = np.random.rand(2, 2)
b1 = np.random.rand(2)
w2 = np.random.rand(2)
b2 = np.random.rand()

hist_cost = []
'''
The training part.
This is a bit jank because everything is from
the top of my noggin.
The idea is to make copies of all of the matrices,
then go through all the parameters one by one, wiggling them
slightly and, as per the definition of the derivative,
go in the opposite direction of the derivative approximation.
'''
for iter in range(20000):
    w1u = copy(w1)
    b1u = copy(b1)
    w2u = copy(w2)
    b2u = copy(b2)
    c = cost(w1, b1, w2, b2)

    # Going through w1
    for i in range(2):
        for j in range(2):
            w1save = w1[i][j]
            w1[i][j] += eps
            w1u[i][j] -= rate*(cost(w1, b1, w2, b2)-c)/eps
            w1[i][j] = w1save

    # Going through b1 and w2, because they
    # are the same dimensions.
    for i in range(2):
        w2save = w2[i]
        w2[i] += eps
        w2u[i] -= rate*(cost(w1, b1, w2, b2)-c)/eps
        w2[i] = w2save

        b1save = b1[i]
        b1[i] += eps
        b1u[i] -= rate*(cost(w1, b1, w2, b2)-c)/eps
        b1[i] = b1save

    # Finally doing b2 (just a float) like last time.
    db2 = (cost(w1, b1, w2, b2 + eps) - c)/eps
    b2 -= rate*db2
    w1 = w1u
    w2 = w2u
    b1 = b1u
    hist_cost = hist_cost + [cost(w1, b1, w2, b2)]

# Printing the results
print(f"Cost: {cost(w1, b1, w2, b2)}")
print("----------------")
print("Input1\tInput2\tResult")
for i in [0, 1]:
    for j in [0, 1]:
        print(f"{i}\t{j}\t{forward([i,j], w1, b1, w2, b2)}")

print(f"Cost: {cost(w1, b1, w2, b2)}")
print("----------------")
print("Input1\tInput2\tResult")
for i in [0, 1]:
    for j in [0, 1]:
        print(f"{i}\t{j}\t{int(forward([i,j], w1, b1, w2, b2)+0.5)}")


# Plotting
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(range(0, len(hist_cost)), hist_cost, c='k')
ax.set(xlabel="Iteration", ylabel="Cost")
plt.show()

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
x_flat = x.flatten()
y_flat = y.flatten()
z_flat = np.array([forward(np.array([i, j]), w1, b1, w2, b2) for i, j in zip(x_flat, y_flat)])
z = z_flat.reshape(x.shape)
ax.plot_surface(x, y, z, alpha=0.25, label="Model")
ax.legend()
plt.show()
