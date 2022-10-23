import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

from Dataset_Preparation import load_dataset, DATA_PATH

import warnings

warnings.filterwarnings("ignore")

PREF_PATH = "performance.txt"
# set 0 to use all data
DATA_SIZE = 100
MNB_PARAM = {'alpha': [0.5, 0.0, 1.0, 10.0]}
DT_PARAM = {'criterion': ['gini', 'entropy'],
            'max_depth': [5, 10],
            'min_samples_split': [2, 4, 6]}
MLP_PARAM = {'hidden_layer_sizes': [(30, 50), (10, 10, 10)],
             'activation': ['logistic', 'tanh', 'relu', 'identity'],
             'solver': ['sgd', 'adam'],
             'max_iter': [5, 10]}


# Process the dataset using feature sklearn.extraction.text.CountVectorizer to extract tokens/words
# and their frequencies. Display the number of tokens (the size of the vocabulary) in the dataset
def extract_features(data):
    vectorizer = CountVectorizer()
    words = [item[0] for item in data]
    count_vector = vectorizer.fit_transform(words)
    return vectorizer, count_vector


def to_frequencies(X_train_counts):
    from sklearn.feature_extraction.text import TfidfTransformer
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    return X_train_tfidf


def split_data(data, tokenize=False, tf_idf=False, count_vector=None, size_of_data=DATA_SIZE):
    from nltk.tokenize import word_tokenize
    if tokenize:
        X = [word_tokenize(item[0]) for item in data]
    else:
        if count_vector is not None:
            if tf_idf:
                X = to_frequencies(count_vector)
            else:
                X = count_vector
        else:
            X = [item[0] for item in data]
    y1 = [item[1] for item in data]
    y2 = [item[2] for item in data]
    if size_of_data == 0:
        size_of_data = len(data)
    return train_test_split(X[:size_of_data], y1[:size_of_data], test_size=0.2, random_state=42), \
           train_test_split(X[:size_of_data], y2[:size_of_data], test_size=0.2, random_state=42)


def train_models(model, X_train, y_train, param_grid=None):
    if param_grid is not None:
        model = GridSearchCV(estimator=model, param_grid=param_grid)
    model.fit(X_train, y_train)
    return model


def classification_task(model, X_test, y_test):
    from sklearn.metrics import confusion_matrix, classification_report
    y_pred = model.predict(X_test)
    matrix = confusion_matrix(y_test, y_pred, labels=np.unique(y_pred))
    report = classification_report(y_test, y_pred, labels=np.unique(y_pred))
    return matrix, report


def save_performance(model, X_test, y_test, data_type: str, path=PREF_PATH):
    matrix, report = classification_task(model, X_test, y_test)
    result = data_type + " " + str(model) + ":" + str(model.get_params()) + "\n"
    result += "Confusion Matrix: \n" + str(matrix) + "\n"
    result += "Classification Report: \n" + str(report) + "\n"
    result += "--------------------------------------------\n"
    print(result)
    with open(path, 'a+') as f:
        f.write(result)


def word(tf_idf=False, path=PREF_PATH):

    data = load_dataset(DATA_PATH)
    vectorizer, count_vector = extract_features(data)
    print("The size of the vocabulary is: ", len(vectorizer.vocabulary_))
    # Split the dataset into 80% for training and 20% for testing

    (X_train, X_test, y_train_e, y_test_e), (t1, t2, y_train_s, y_test_s) = split_data(data, tf_idf=tf_idf, count_vector=count_vector)
    # X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(X_train[0:100], emotion[0:100], test_size=0.2,
    #                                                             random_state=0)
    # X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_train[0:100], sentiment[0:100], test_size=0.2,
    #                                                             random_state=0)

    # Train a Multinomial Naive Bayes classifier on the training set

    models = [DecisionTreeClassifier(), MultinomialNB(), MLPClassifier(max_iter=1)]
    with open(path, "w") as f:
        f.write("")
    for model in models:
        emotion_model = train_models(model, X_train, y_train_e)
        save_performance(emotion_model, X_test, y_test_e, "Emotion",path=path)
        sentiment_model = train_models(model, X_train, y_train_s)
        save_performance(sentiment_model, X_test, y_test_s, "Sentiment",path=path)
        print("Model trained: ", str(model))
        if str(model) == "MultinomialNB()":
            emotion_top_model = train_models(model, X_train, y_train_e, MNB_PARAM)
            sentiment_top_model = train_models(model, X_train, y_train_s, MNB_PARAM)
        elif str(model) == "DecisionTreeClassifier()":
            emotion_top_model = train_models(model, X_train, y_train_e, DT_PARAM)
            sentiment_top_model = train_models(model, X_train, y_train_s, DT_PARAM)
        elif str(model) == "MLPClassifier()":
            emotion_top_model = train_models(model, X_train, y_train_e, MLP_PARAM)
            sentiment_top_model = train_models(model, X_train, y_train_s, MLP_PARAM)
        else:
            print("Model not found")
            break
        save_performance(emotion_top_model, X_test, y_test_e, "Emotion", path=path)
        save_performance(sentiment_top_model, X_test, y_test_s, "Sentiment", path=path)
        print("GridSearchCV trained: ", str(model))

    print("Done!")


if __name__ == "__main__":
    word(path=PREF_PATH)
    # Use tf-idf instead of word frequencies and redo all substeps of 2.3 above – you can use TfidfTransformer
    # for this. Display the results of this experiment

    word(tf_idf=True, path="tf_performance.txt")
