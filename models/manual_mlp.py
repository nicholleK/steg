import pickle

import numpy as np

class ManualMLP:
    # manual implementation of MLP

    def __init__(
        self,
        layer_sizes,
        learning_rate=0.001,
        random_state=42

    ):

        np.random.seed(random_state) # set random seed
        # store network architecture
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate # lr for gradient descent
        self.weights = []
        self.biases = []
        self.loss_history =[] # for UI display
        self.accuracy_history = [] # store accuracy after each epoch
        self.initialise_parameters() # for weights and biases init


    def initialise_parameters(self):
        # creates and initializes all weights and biases vectors

        # clear both for reinitialization
        self.weights = []
        self.biases = []

        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i] # store the num of inputs going into a layer
            fan_out = self.layer_sizes[i + 1] # store the num of outputs going out of a layer
            #  create weights matrix
            weight = (
                np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in) # using He init
            )

#             initialise every bias to 0
            bias = np.zeros((1, fan_out)) # [1, num_neurons]
#             store weight matrix
            self.weights.append(weight)
#             store bias vector
            self.biases.append(bias)

    def relu(self, x):
        # ReLU activation func - the max(0,x)
        return np.maximum(0, x)

    def relu_derivative(self, x):
        # ReLU derivative for backprop - x>0, deriv = 1, else 0
        return (x > 0).astype(np.float32) # return boolean array

    def softmax(self, x):
#         converts logits into probs
        x = x - np.max(x, axis=1, keepdims=True) # simplify logits for conversion

#         calc exponential for every element
        exp = np.exp(x)

        return exp / np.sum(exp, axis=1, keepdims=True)  # convert the logits into probs

    def one_hot(self, labels):
        # used to enable comparison between true labels and prob outputs
        n_classes = self.layer_sizes[-1]
        encoded = np.zeros((len(labels), n_classes))
        encoded[np.arange(len(labels)), labels] = 1 # input 1 in the correct position
        return encoded # return the encoded matrix

    def cross_entropy_loss(self, probs, labels):
#         calc avg CEL

        t = len(labels)
        # validate against log(0)
        probs = np.clip(probs, 1e-12, 1.0)

        # select pred prob vs correct class
        loss = -np.log(probs[np.arange(t), labels])
        return np.mean(loss)

    def forward_propagation(self, X):
        # X - input feature matrix
        # shape - (batch_size, num_features)

        self.activations = []
        self.weighted_sums = []

        current = X
        self.activations.append(current)

        for i in range (len(self.weights)):

            # matrix multiplication - Z = A (prev activations) W (weights) + b (bias)

            Z = np.dot(current, self.weights[i]) + self.biases[i]
            self.weighted_sums.append(Z) # save for backprop


            if i < len(self.weights) - 1:
                current = self.relu(Z) # hidden layer - use ReLU
            else:
                current = self.softmax(Z) # output layer - use softmax

            self.activations.append(current) # save activation

        return current # softmx probs


    def backward(self, Y_true):
#         Y_true - array of true class labels
#         updates weights & biases using gradient descent

        t = len(Y_true) # store num training samples in batch
        one_hot = self.one_hot(Y_true) # convert labels into 1-hot vectors

        delta = self.activations[-1] - one_hot # delta  = pred probs - true labels (one_hot's conversion)

        #  working backwards through each layer
        for layer in reversed(range(len(self.weights))):
            grad_weights = np.dot(self.activations[layer].T, delta) / t
            grad_biases = np.sum(delta, axis=0, keepdims=True) / t

            grad_weights = np.clip(grad_weights, -5.0, 5.0)
            grad_biases = np.clip(grad_biases, -5.0, 5.0)

            old_weights = self.weights[layer].copy()

            # gradient descent calc: W = W -lr . grad_weights
            self.weights[layer] -= (self.learning_rate * grad_weights) # Update weights
            # B = B -lr . grad_biases
            self.biases[layer] -= (self.learning_rate * grad_biases)  #  Update biases


#             compute error for prev layer if its not 1st layer
            if layer > 0:
                delta = np.dot(delta, old_weights.T)
                delta *= self.relu_derivative(self.weighted_sums[layer - 1])


    def fit(self, X_train, Y_train, epochs=50, batch_size=32, verbose=True):
        # responsible for training the n-network using mini-batch grad descent

        # X_train - training feature array
        # Y_train - training labels array
        # epochs - int of complete training passes
        # batch_size - num samples used before weights update
        # verbose - print progress

        n_samples = len(X_train) # total # of training samples

        for epoch in range(epochs):
            idx = np.random.permutation(n_samples) # randomize training data to prevent bias learning
            X_train = X_train[idx]
            Y_train = Y_train[idx]

            epoch_loss = 0.0 # storage

#            process each batch at a time
            for start_idx in range(0, n_samples, batch_size):
                end_idx = start_idx + batch_size # get end index of current batch
                # get mini batch
                batch_X = X_train[start_idx:end_idx]
                batch_Y = Y_train[start_idx:end_idx]

                # forward prop to get softmax probs
                probs = self.forward_propagation(batch_X)

                # Calc CEL
                loss = self.cross_entropy_loss(probs, batch_Y)
                epoch_loss += loss # add current batch loss to total loss

                # back prop to update weights
                self.backward(batch_Y)

            # calc avg loss across all batches
            avg_loss = epoch_loss / len(range(0, n_samples ,batch_size))

            # store loss for UI plotting
            self.loss_history.append(avg_loss)

            # Calc training accuracy
            predictions = self.predict(X_train) # store predicted class labels
            accuracy = np.mean(predictions == Y_train)

            # save accuracy history
            self.accuracy_history.append(accuracy)

            if verbose:
                print(f"Epoch "
                      f"{epoch+1}/{epochs}, "
                      f" | "
                      f"Loss: {avg_loss:.4f}, "
                      f" | "
                      f"Accuracy: {accuracy:.4f}, "
                      )
        return {"loss_history": self.loss_history, "accuracy_history": self.accuracy_history}

    def predict_proba(self, X):
        # predicts the probs of every class

        # X - feature matrix
        # returns array of predicted probs
        probs = self.forward_propagation(X)
        return  probs


    def predict(self, X):
        # predicts the class label

        # X - feature matrix
        # returns idx of predicted class
        probs = self.predict_proba(X) # get the predicted probabilities
        predictions = np.argmax(probs, axis=1) # select class with highest probability
        return  predictions # returns the indices

    def evaluate(self, X_test, Y_test):
        # evaluates & returns classificatio accuracy
        predictions = self.predict(X_test) # predict every image
        accuracy = np.mean(predictions == Y_test)  # compare prediction against correct label
        return accuracy










