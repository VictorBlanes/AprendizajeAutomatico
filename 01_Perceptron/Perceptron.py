import numpy as np


class Perceptron:
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting. w[0] = threshold
    errors_ : list
        Number of miss classifications in each epoch.

    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        self.w_ = None  # defined in method fit

    def fit(self, X, y):

        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        """
        self.w_ = np.zeros(1 + X.shape[1])  # First position corresponds to threshold

        for _ in range(self.n_iter):
            for dataSample, target in zip(X, y):
                # Se calcula la prediccion con el resultado de la funcion de activacion y no con el producto escalar 
                # del data sample con los pesos
                calculatedPrediction = 1 if np.dot(dataSample, self.w_[1:]) + self.w_[0] >= 0.0 else -1
                delta_w = self.eta * (target - calculatedPrediction)
                # Se cambia el bias con el ratio de update de los pesos because reasons
                self.w_[0] += delta_w
                self.w_[1:] += delta_w * dataSample[:]

    def predict(self, X):
        """Return class label.
            First calculate the output: (X * weights) + threshold
            Second apply the step function
            Return a list with classes
        """

        predictions = np.dot(X, self.w_[1:])
        return np.where(predictions >= 0, 1, -1)
