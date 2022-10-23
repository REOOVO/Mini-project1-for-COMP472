import numpy as np
from nltk import word_tokenize
from sklearn.neural_network import MLPClassifier

import gensim.downloader as api

from Dataset_Preparation import load_dataset, DATA_PATH

import warnings

from Words_as_Features import train_models, save_performance, split_data

warnings.filterwarnings("ignore")


def get_mean_vector_conceptnet(word2vec_model, words):
    # remove out-of-vocabulary words
    words = [('/c/en/' + word) for word in words if ('/c/en/' + word) in word2vec_model.wv.vocab.keys()]
    if len(words) >= 1:
        return np.mean(word2vec_model[words], axis=0)
    else:
        return []


def sent_vectorizer_conceptnet(sent, model):
    sent_vec = []
    numw = 0
    for w in sent:
        try:
            w = '/c/en/' + w
            if numw == 0:
                sent_vec = model.wv[w]
            else:
                sent_vec = np.add(sent_vec, model[w])
            numw += 1
        except:
            pass

    return np.asarray(sent_vec) / numw


def data_vectorizer_conceptnet(sents, model):
    data_vec = []
    for sent in sents:
        data_vec.append(sent_vectorizer_conceptnet(sent, model))
    return data_vec


def get_post_hit_rate__conceptnet(model, set):
    hits = [1 for i in set if ('/c/en/' + i) in model.vocab.keys()]
    return sum(hits) / len(set)


if __name__ == "__main__":
    vector = api.load("conceptnet-numberbatch-17-06-300")
    path = 'conceptnet.txt'
    size = 0

    print(vector.vector_size)
    datas = load_dataset(DATA_PATH)
    # clean data
    data = []
    for value in datas:
        words = word_tokenize(value[0])
        if len(get_mean_vector_conceptnet(vector, words)) >= 1:
            data.append(value)
    print("total remove:" + str(len(datas) - len(data)))

    # 3.2
    (X_train, X_test, y_train_e, y_test_e), (_, _, y_train_s, y_test_s) = split_data(data, tokenize=True,
                                                                                     size_of_data=size)

    print("The size of the tokens is: ", len(X_train))
    # Compute the embedding of a Reddit post as the average of the embeddings of its words. If
    # a word has no embedding in Word2Vec, skip it.
    sentence_embeddings = get_mean_vector_conceptnet(vector, X_train[0])
    average = sum(sentence_embeddings) / len(sentence_embeddings)
    print(average)
    # 3.3
    embeddings_train = []
    for post in X_train:
        embedding = get_mean_vector_conceptnet(vector, post)
        embeddings_train.append(sum(embedding) / len(embedding) if len(embedding) != 0 else 0)

    print(embeddings_train[0:100])

    # 3.4
    # Compute and display the overall hit rates of the training and test sets
    train_hit_rates = []
    for post in X_train[0:100]:
        train_hit_rates.append(get_post_hit_rate__conceptnet(vector, post))
    print("The hit rate of the training set is: ", sum(train_hit_rates) / len(train_hit_rates))
    test_hit_rates = []
    for post in X_test[0:100]:
        test_hit_rates.append(get_post_hit_rate__conceptnet(vector, post))
    print("The hit rate of the test set is: ", sum(test_hit_rates) / len(test_hit_rates))

    # 3.5

    X_train = data_vectorizer_conceptnet(X_train, vector)
    X_test = data_vectorizer_conceptnet(X_test, vector)
    # X_train = embeddings_train
    # X_test = embeddings_test
    # test = 0
    # for value in X_train:
    #   if test == 0:
    #     test = value.shape
    #   elif value.shape != test:
    #     print(value.shape)
    #     break
    with open(path, "w") as f:
        f.write("")
    model = MLPClassifier(max_iter=1)
    base_MLP_e = train_models(model, X_train, y_train_e)
    save_performance(base_MLP_e, X_test, y_test_e, "base_MLP_e", path=path)
    base_MLP_s = train_models(model, X_train, y_train_s)
    save_performance(base_MLP_s, X_test, y_test_s, "base_MLP_s", path=path)
    # print(type(base_MLP_e), type(top_MLP_e), type(base_MLP_s), type(top_MLP_s))
