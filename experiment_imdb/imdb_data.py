from keras.datasets import imdb
from gensim.models import Word2Vec
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
import numpy as np
import gc
from sys import getsizeof
from time import sleep


PATH_W2V_SG="/media/ghy/study/nlp/data/Word2VecModels/model_sg.bin"


def load_imdb_data():
    (X_train, y_train), (X_test, y_test) = imdb.load_data(path="imdb_full.pkl",
                                                      nb_words=None,
                                                      skip_top=0,
                                                      maxlen=None,
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)

    # Converting labels to binary vectors
    y_train_cat = to_categorical(y_train, nb_classes=2)
    y_test_cat = to_categorical(y_test, nb_classes=2)


    del y_train
    del y_test
    gc.collect()


    data = imdb.get_word_index("imdb_word_index.pkl")
    data_reverse = dict(zip(data.values(),data.keys()))

    del data
    gc.collect()


    model_sg = Word2Vec.load(PATH_W2V_SG)

    NUM_TOKEN = 200
    X_train_list=[]
    X_test_list=[]
    embedding_size=200 ##both models have this size
    for tokens in X_train:
        w2v=[]
        for i in range(0,min(NUM_TOKEN,len(tokens))):
            try:
                token_word = data_reverse[tokens[i]]
                token_vector=model_sg[token_word]
                w2v.append(token_vector)
            except:
                w2v.append(embedding_size*[0])
        while(i<NUM_TOKEN-1):
            w2v.append(embedding_size*[0])
            i+=1

        #w2v = np.array(w2v)
        X_train_list.append(w2v)

    del X_train
    gc.collect()


    for tokens in X_test:
        w2v=[]
        for i in range(0,min(NUM_TOKEN,len(tokens))):
            try:
                token_word = data_reverse[tokens[i]]
                token_vector=model_sg[token_word]
                w2v.append(token_vector)
            except:
                w2v.append(embedding_size*[0])
        while(i<NUM_TOKEN-1):
            w2v.append(embedding_size*[0])
            i+=1
        X_test_list.append(w2v)


    del model_sg
    del X_test
    gc.collect()

    return X_train_list,X_test_list,y_train_cat,y_test_cat





