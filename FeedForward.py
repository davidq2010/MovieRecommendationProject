import numpy as np
from sklearn.neural_network import MLPClassifier

# Define the function behind the dataset
def get_target(x):
    return 1 \
        if (x[0] > 0.5 and x[1] <= 0.5) \
           or (x[0] <= 0.5 and x[1] > 0.5) \
        else 0


def generate_dataset(size):
    # Spread the inputs around the corners of [0,1]^2 instead of uniformly
    inputs = np.random.randint(0, 2, (size, 2))
    inputs = inputs.astype(float)
    # Add some noise to the points on the edges
    inputs += np.random.randn(size, 2) / 10
    targets = np.apply_along_axis(get_target, 1, inputs)
    return inputs, targets


def activation(x):
    return 1 / (1 + np.exp(-x))


def d_activation(x):
    return activation(x) * (1 - activation(x))


# Generate a dataset
inputs, targets = generate_dataset(200)

n_hidden = 2

W_ih = np.random.randn(2, n_hidden)  # Weights between input and hidden layer
W_ho = np.random.rand(n_hidden, 1)  # Weights between hidden and output layer
b_h = np.random.randn(1, n_hidden)  # Biases of the hidden layer
b_o = np.random.rand(1, 1)  # Bias of the output layer

mse = float("inf")
misclassifications = 0
epoch = 1
learning_rate = 0.1

while mse > 1e-6 and epoch < 2000:
    # Feed each input vector to the neuron and keep the errors for calculating
    # the MSE.
    errors = []
    misclassifications = 0
    for x, y in zip(inputs, targets):
        # Bring x into the right shape for the vector-matrix multiplication
        x = np.reshape(x, (1, 2))
        # Feed x to every neuron in the hidden layer by multiplying it with
        # their weight vectors
        net_h = x.dot(W_ih) + b_h
        out_h = activation(net_h)
        net_o = out_h.dot(W_ho) + b_o
        prediction = activation(net_o)

        # Evaluate the quality of the net
        label = 1 if prediction > 0.5 else 0
        if label != y:
            misclassifications += 1

        # Calculate the error
        error = np.square(y - prediction)
        errors.append(error)

        # Adjust the weights and the bias by calculating the gradient
        # Weight updates in the output layer
        delta_o = (y - prediction) * d_activation(net_o)
        W_ho += learning_rate * delta_o * out_h.T
        b_o += learning_rate * delta_o

        # Weight updates in the hidden layer
        delta_h = d_activation(net_h) * W_ho.T * delta_o
        W_ih += learning_rate * x.T.dot(delta_h)
        b_h += learning_rate * delta_h

    # Calculate the mean squared error
    new_mse = np.mean(errors)
    if epoch % 10 == 1:
        print "Error after epoch %d: %s" % (epoch, new_mse)

    # If the error did not improve, decrease the learning rate, since we
    # might be close to an optimum
    if new_mse > mse and epoch > 20:
        learning_rate *= 0.5
        learning_rate = max(learning_rate, 1e-6)
        print "New learning rate: %s" % learning_rate

    # If the error converges (possibly in a local minimum), stop the training
    if 0 < mse - new_mse < 1e-6:
        break

    mse = new_mse
    epoch += 1

print "Finished training after %d epochs. Misclassifications in last epoch: " \
      "%f%%" % (epoch, misclassifications * 100.0 / len(inputs))


#clf = MLPClassifier(solver='sgd', activation='logistic',alpha=1e-5, learning_rate_init = 0.1,hidden_layer_sizes=(2,))
clf = MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(2,))
clf.fit(inputs, targets)        
yhats = clf.predict(inputs)
print(clf.score(inputs,targets))


