from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from logisticRegression import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_ROC(p, y, state='scatter'):
    """
    This function plot ROC curve given predicted probabilities and true binary labels
    :param p: (numpy array): predicted probabilities of positive class (shape: [n_samples,])
    :param y: (numpy array): true binary labels (shape: [n_samples,])
    :param state: (str): plot state, either 'scatter' or 'info' or 'line'
    :return: None
    """
    Ts = np.linspace(0, 1, 20)
    pts = []
    for T in Ts:
        y_pred = (p >= T).astype(int)  # Apply decision threshold and convert to binary predictions
        y[y == -1] = 0  # Convert -1 labels to 0
        y = y.astype(int)
        # Compute confusion matrix and FPR/TPR
        E = confusion_matrix(y, y_pred)
        pts.append((E[0, 1] / (E[0, 1] + E[0, 0]), E[1, 1] / (E[1, 0] + E[1, 1])))
        if state == 'info':
            print(E)
            print("True Positive Ration: ", E[1, 1] / (E[1, 0] + E[1, 1]))
            print("True Positive Ration: ", E[0, 1] / (E[0, 1] + E[0, 0]))

    # Plot ROC curve
    xx, yy = zip(*pts)
    if state == 'scatter':
        plt.scatter(xx, yy, s=20, color='black')
        plt.plot(xx, yy, c='red')

    x = np.linspace(0, 1, 20)
    plt.plot(x, x, color='green', linestyle=':')
    plt.yticks([0, 1], ['0', '1'])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True\n Positive\n Rate\n (Recall)', rotation='horizontal')
    plt.show()


def plot_ROC_from_clf(clf, X, y, state='scatter'):
    """
    This function plot ROC curve for a classifier given input features and true binary labels.
    :param clf: a classifier object with predict_proba method
    :param X: (numpy array) input features (shape: [n_samples, n_features])
    :param y: (numpy array): true binary labels (shape: [n_samples,])
    :param state: plot state, either 'scatter' or 'info' or 'line'
    :return: None
    """
    p = clf.predict_proba(X)
    plot_ROC(p, y, state)


if __name__ == '__main__':
    # Load dataset
    data = pd.read_csv('spam_ham_dataset.csv')
    texts = data['text']
    label = np.array(data['label'].copy())
    label[label == 'ham'] = -1
    label[label == 'spam'] = 1

    # create vectorazier
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    # Compute the mean tf-idf score for each word across the training set
    tfidf_means = np.asarray(X.mean(axis=0)).ravel()
    top_indices = tfidf_means.argsort()[-1000:][::-1]  # Get indices of top 1000 words
    relevant_word_matrix = X[:, top_indices]

    # split the data to train and test
    X_train, X_test, y_train, y_test = train_test_split(relevant_word_matrix, label, test_size=0.2, random_state=42)

    # train the model and plot
    model = LogisticRegression()
    model.fit(X_train, y_train)
    print("The score of the model is {}".format(model.score(X_test, y_test)))
    print("The weights vector is {}".format(model.weights_))
    plot_ROC_from_clf(model, X_test, y_test)
