import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3, loss=[]):
        """Linear Regression using Gradient Descent.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.weights = None
        self.bias = None

    def fit(self, X, y, batch_size=32, regularization=0, max_epochs=100, patience=3, alpha=0.01):
        """Fit a linear model.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience

        # TODO: Initialize the weights and bias based on the shape of X and y.
        # weights have a normal distribution and a standard deviation of 0.01
        self.weights = np.random.normal(0, 0.01, size=(X.shape[1], y.shape[1]))
        self.bias = np.zeros((1, y.shape[1]))
        # Check that X and y have the same number of samples(learnt this the hard way)
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "The number of samples in X and y must be the same")
        num_samples = int(X.shape[0])
        # assigning the 1st 90% of the data to x and y train variables
        X_train = X[:int(0.9 * num_samples)]
        y_train = y[:int(0.9 * num_samples)]
        # assigning the last 10% for validation
        val_X = X[int(0.9 * num_samples):]
        val_y = y[int(0.9 * num_samples):]
        num_train_samples = X_train.shape[0]

        # for adjusting the weights and bias, we use: batch gradient descent
        error_store = []
        training_loss = []
        compare = 0
        old_loss = float('inf')
        
        for epoch in range(self.max_epochs):
            for i in range(0, num_train_samples, self.batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                y_predicted = np.dot(X_batch, self.weights) + self.bias
                error = y_predicted - y_batch
                prev_loss = np.mean(error**2)
                gradient = X_batch.T @ error / batch_size
                gradient = gradient + 2 * regularization * self.weights
                self.weights = self.weights - alpha * gradient
                grad_bias = (1 / batch_size) * np.sum(error)
                self.bias -= grad_bias

                # Evaluate the model on the validation set after each iteration
                y_predicted_val = np.dot(val_X, self.weights)
                val_error = y_predicted_val - val_y
                validation_loss = np.mean(val_error**2)
                # print("Validation loss:", validation_loss)
                # print("Previous Loss", prev_loss)
                training_loss.append(prev_loss)
                # early stopping
                error_store.append(validation_loss)

                if prev_loss < old_loss:
                    old_loss = prev_loss
                    weight = self.weights
                    grad_bias = self.bias
                    compare = 0

                else:
                    compare += 1
                    if compare >= patience:
                        self.weights = weight
                        self.bias = grad_bias
                        print(f'Early stopping at epoch {epoch}')
                        # print("error store for validation", error_store)
                        # print(" training loss for training data", training_loss)
                        #print("counter", compare)
                        break
                    break
                break
            self.weights = weight.copy()
            self.bias = grad_bias.copy()
        # print("weights",self.weights)
        # print("bias",self.bias)
        plt.plot(range(len(training_loss)), training_loss)
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Loss curve")
        plt.show()

        # TODO: Implement the training loop.

    def predict(self, X):
        """Predict using the linear model.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        """
        # TODO: Implement the prediction function.
        return X.dot(self.weights) + self.bias

    def score(self, X, y):
        """Evaluate the linear model using the mean squared error.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target data.
        """
        # TODO: Implement the scoring function.
        y_predicted = self.predict(X)
        return np.mean((y_predicted - y) ** 2)
