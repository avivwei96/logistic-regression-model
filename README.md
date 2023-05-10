# Logistic Regression Model

This is a simple implementation of the Logistic Regression algorithm in Python, using numpy library.

The `LogisticRegression` class contains the following methods:

- `fit(data, labels)`: fits the model to the data and labels.
- `_sigmoid_func(X)`: applies the sigmoid function to the product of `X` and the weights of the model.
- `cross_entropy_grad(X, labels)`: calculates the gradient of the loss function with respect to the weights.
- `predict_proba(data)`: predicts the probability of the data being labeled as 1.
- `predict(data)`: predicts the label of the data.
- `score(data, labels)`: calculates the accuracy score of the model on the given data and labels.

## Usage

To use this model, first import the class:

```python
from logistic_regression import LogisticRegression
```

Then, create an instance of the class:

```python
model = LogisticRegression()
```

You can fit the model to your data by calling the `fit` method:

```python
model.fit(X_train, y_train)
```

After fitting the model, you can predict the labels of new data by calling the `predict` method:

```python
y_pred = model.predict(X_test)
```

You can also calculate the accuracy score of the model by calling the `score` method:

```python
score = model.score(X_test, y_test)
```

## Example Usage

The `spam_ham_dataset.csv` file contains a dataset of email messages labeled as spam or ham (non-spam). We will use this dataset to train a logistic regression model to predict whether a given email message is spam or ham.

First, we use the `TfidfVectorizer` from `sklearn.feature_extraction.text` to create a feature matrix from the email messages. We then select the top 1000 words based on their mean tf-idf score across the training set, and use these words as features for our logistic regression model.

We split the data into a training set and a test set, with 80% of the data used for training and 20% used for testing. We then train a logistic regression model on the training set using our selected features.

Finally, we evaluate the performance of the model on the test set by computing its accuracy and plotting the ROC curve. We also print the weights vector learned by the logistic regression model.

To run the code, simply execute the `main.py` file.
