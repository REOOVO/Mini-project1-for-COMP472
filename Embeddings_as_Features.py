import gensim.downloader as api
import numpy as np
from gensim.models import Word2Vec
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

from Dataset_Preparation import load_dataset, DATA_PATH
from Words_as_Features import split_data, train_models, save_performance, MLP_PARAM, DATA_SIZE


def get_mean_vector(word2vec_model,  words):
    # remove out-of-vocabulary words
    words = [word for word in words if word in word2vec_model.index_to_key]
    if len(words) >= 1:
        return np.mean(word2vec_model[words], axis=0)
    else:
        return []


def get_post_hit_rate(model, set):
    hits = [1 for i in set if i in model.index_to_key]
    return sum(hits)/len(set)


def sent_vectorizer(sent, model):
    sent_vec = []
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                sent_vec = model[w]
            else:
                sent_vec = np.add(sent_vec, model[w])
            numw += 1
        except:
            pass

    return np.asarray(sent_vec) / numw


def data_vectorizer(sents, model):
    data_vec = []
    for sent in sents:
        data_vec.append(sent_vectorizer(sent, model))
    return data_vec


def main(path, vectors=None, size=DATA_SIZE):

    print(vectors.vector_size)
    datas = load_dataset(DATA_PATH)
    # clean data
    data = []
    for value in datas:
        words = word_tokenize(value[0])
        if len(get_mean_vector(vectors, words)) >= 1:
            data.append(value)
    print("total remove:" + str(len(datas) - len(data)))
    # 3.2
    (X_train, X_test, y_train_e, y_test_e), (_, _, y_train_s, y_test_s) = split_data(data, tokenize=True, size_of_data=size)

    print("The size of the tokens is: ", len(X_train))
    # Compute the embedding of a Reddit post as the average of the embeddings of its words. If
    # a word has no embedding in Word2Vec, skip it.
    sentence_embeddings = get_mean_vector(vectors, X_train[0])
    average = sum(sentence_embeddings)/len(sentence_embeddings)
    print(average)
    # 3.3
    embeddings = []
    for post in X_train[0:100]:
        embedding = get_mean_vector(vectors, post)
        embeddings.append(sum(embedding)/len(embedding))

    print(embeddings)

    # 3.4
    # Compute and display the overall hit rates of the training and test sets
    train_hit_rates = []
    for post in X_train[0:100]:
        train_hit_rates.append(get_post_hit_rate(vectors, post))
    print("The hit rate of the training set is: ", sum(train_hit_rates)/len(train_hit_rates))
    test_hit_rates = []
    for post in X_test[0:100]:
        test_hit_rates.append(get_post_hit_rate(vectors, post))
    print("The hit rate of the test set is: ", sum(test_hit_rates)/len(test_hit_rates))

    # 3.5

    X_train = data_vectorizer(X_train, vectors)
    X_test = data_vectorizer(X_test, vectors)

    base_MLP_e = train_models(MLPClassifier(), X_train, y_train_e)
    base_MLP_s = train_models(MLPClassifier(), X_train, y_train_s)
    top_MLP_e = train_models(MLPClassifier(), X_train, y_train_e, param_grid=MLP_PARAM)
    top_MLP_s = train_models(MLPClassifier(), X_train, y_train_s, param_grid=MLP_PARAM)
    # print(type(base_MLP_e), type(top_MLP_e), type(base_MLP_s), type(top_MLP_s))
    with open(path, "w") as f:
        f.write("")
    save_performance(base_MLP_e, X_test, y_test_e, "base_MLP_e", path)
    save_performance(top_MLP_e, X_test, y_test_e, "top_MLP_e", path)
    save_performance(base_MLP_s, X_test, y_test_s, "base_MLP_s", path)
    save_performance(top_MLP_s, X_test, y_test_s, "top_MLP_s", path)


if __name__ == "__main__":
    vectors = api.load("word2vec-google-news-300")
    main("embedding.txt", vectors, size=100)
