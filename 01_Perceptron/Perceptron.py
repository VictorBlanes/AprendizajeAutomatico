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
            for indexSample, dataSample in enumerate(X):
                calculatedPrediction = 0
                for indexFeature, singleFeature in enumerate(dataSample):
                    calculatedPrediction += singleFeature * self.w_[indexFeature + 1]
                for indexWeight in range(X.shape[1]):
                    weightVariation = self.eta*(y[indexSample] - calculatedPrediction)*X[indexSample][indexWeight]
                    self.w_[indexWeight + 1] += weightVariation

    def predict(self, X):
        """Return class label.
            First calculate the output: (X * weights) + threshold
            Second apply the step function
            Return a list with classes
        """

        predictions = np.zeros(X.shape[0])
        for predictionIndex in range(predictions.shape[0]):
            prediction = 0
            for indexPredictionValue, predictionValue in enumerate(X[predictionIndex]):
                prediction += predictionValue*self.w_[indexPredictionValue + 1]
            if prediction >= 0:
                prediction = 1
            else:
                prediction = -1
            predictions[predictionIndex] = prediction
        return predictions
