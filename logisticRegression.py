import numpy as np


class LogisticRegression(object):
    def fit(self, data, labels):
        """
        this function fit the model to the data and the labels
        :param data: numeric data array
        :param labels 1 and -1
        :return: None
        """
        num_of_cols = data.shape[1]
        self.weights_ = np.random.rand(num_of_cols)
        self.lr = 0.1
        self.epochs = 200
        for epoch in range(self.epochs):
            self.cross_entropy_grad(data, labels)
            self.weights_ -= self.lr * self.grad

    def _sigmoid_func(self, X):
        """
        This function does the sigmoid function of (X * the weights of the model)
        :param X: Vector size of th weights
        :return: The result of the sigmoid function
        """
        return 1 / (1 + np.exp(-X@self.weights_))

    def cross_entropy_grad(self, X, labels):
        """
        This function calculates the gradient of the loss function due to the weights
        :param X: The numeric features of the data
        :param labels: Labels with the values of {-1,1}
        :return: None
        """
        self.grad = 0
        for vec, label in zip(X, labels):
            self.grad += self._sigmoid_func(vec * -label) * (vec * -label)

    def predict_proba(self, data):
        """
        This function predict the probabilty of the data to be labeled 1
        :param data: Numeric features
        :return: Vector of the predictions
        """
        prediction = self._sigmoid_func(data)
        return prediction

    def predict(self, data):
        """
        This function predict the label of the data
        :param data: Numeric features
        :return: the label prediction
        """
        self.treshold = 0.6
        pred = self.predict_proba(data)
        pred[pred < self.treshold] = -1
        pred[pred >= self.treshold] = 1
        return pred

    def score(self, data, labels):
        """
        this function calculates the score of the model due to the given labels
        :param data: Numeric features
        :param labels: Vec of labels with the values of {-1,1}
        :return: The score of the model
        """
        preds = self.predict(data)
        counter = 0
        for ind, pred in enumerate(preds):
            if pred == labels[ind]:
                counter += 1

        return counter / len(preds)
