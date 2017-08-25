# -*- coding: UTF-8 -*-
import numpy as np
from Prepare import *
from keras.utils import np_utils
from gensim.models import Word2Vec
import pickle
import codecs
# import itertools
# from collections import Counter

PATH_A_2016="data/A_All.tsv"
PATH_B_2016="data/BD_All.tsv"
PATH_DATA_2013="data/2013_data.tsv"
PATH_TEST_DATA_A="data/SemEval2016-task4-test.subtask-A.txt"
PATH_TEST_DATA_B="data/SemEval2016-task4-test.subtask-BD.txt"
PATH_SENT140="data/sent140.txt"
PATH_W2V_SG="/media/ghy/study/nlp/data/Word2VecModels/model_sg.bin"
PATH_W2V_CBW="/media/ghy/study/nlp/data/Word2VecModels/model_cbw.bin"

#PATH_W2V_SG="F:/nlp/data/Word2VecModels/model_sg.bin"
#PATH_W2V_CBW="F:/nlp/data/Word2VecModels/model_cbw.bin"

PATH_SENTIMENT_EMBEDDING="embeddings_sentiment.p"


NUM_TOKEN=20
SEED=1234
def load_a(test_split=0.2,cnn=True, reverse=False,sentiment_embeddings=False,Load_Test=False):

    # print "Loading and Preparing Data..."
    p1=prepare_tweets(PATH_A_2016)
    p2=prepare_tweets(PATH_DATA_2013)
    p1.prepare()
    p2.prepare()
    if(Load_Test):
        p3=prepare_tweets(PATH_TEST_DATA)
        p3.prepare()
        tweets_test=p3.tweets
    tweets=np.concatenate((p1.tweets,p2.tweets),axis=0);
    tag=np.concatenate((p1.tag,p2.tag),axis=0);
    result=[]

    # print "Loading Word2Vec Models and Preparing Data ..."
    embedding_size=200 ##both models have this size
    model_sg = Word2Vec.load(PATH_W2V_SG)
    model_sentiment = pickle.load( open( PATH_SENTIMENT_EMBEDDING, "rb" ) )
    for tweet in tweets:
        tokens=tweet.split()
        w2v=[]
        for i in range(0,min(NUM_TOKEN,len(tokens))):
            try:
                if(sentiment_embeddings):
                    token_vector=model_sentiment[tokens[i]]
                else:
                    token_vector=model_sg[tokens[i]]
                w2v.append(token_vector)
            except:
                w2v.append(np.array(embedding_size*[0]))

        while(i<NUM_TOKEN-1):
            w2v.append(np.array(embedding_size*[0]))
            i+=1

        w2v = np.array(w2v)
        if cnn:
            result.append(np.array([w2v]))
        else:
            result.append(np.array(w2v))

    X=np.array(result)

    result=[]
    if(Load_Test):
        for tweet in tweets_test:
            tokens=tweet.split()
            w2v=[]
            for i in range(0,min(NUM_TOKEN,len(tokens))):
                try:
                    if(sentiment_embeddings):
                        token_vector=model_sentiment[tokens[i]]
                    else:
                        token_vector=model_sg[tokens[i]]
                    w2v.append(token_vector)
                except:
                    w2v.append(np.array(embedding_size*[0]))

            while(i<NUM_TOKEN-1):
                w2v.append(np.array(embedding_size*[0]))
                i+=1
            w2v = np.array(w2v)
            if cnn:
                result.append(np.array([w2v]))
            else:
                result.append(np.array(w2v))

        X_train=X
        y_train=tag

        X_test=np.array(result)
        y_test=None

        y_train= np_utils.to_categorical(y_train, 3)

    else:
        np.random.seed(SEED)
        np.random.shuffle(X)
        np.random.seed(SEED)
        np.random.shuffle(tag)

        # print ("negative = %d") % np.sum(tag==0)
        # print ("positive = %d") % np.sum(tag==1)
        # print ("neutral = %d") % np.sum(tag==2)
        #

        X_train = X[:int(len(X)*(1-test_split))]
        y_train = tag[:int(len(X)*(1-test_split))]

        X_test = X[int(len(X)*(1-test_split)):]
        y_test = tag[int(len(X)*(1-test_split)):]

        y_train= np_utils.to_categorical(y_train, 3)
        y_test= np_utils.to_categorical(y_test, 3)

    # print(X_train.shape)
    return (X_train,y_train),(X_test,y_test)

def Evaluate_Result(pred, y_test):
    # Weighted average of accuracy and f1
    (true_predict, support, f1, recall) = (list(), list(), list(),list())
    for l in [0,1]:
        support.append(np.sum(y_test == l) / float(y_test.size))
        tp = float(np.sum(pred[y_test == l] == l))
        fp = float(np.sum(pred[y_test != l] == l))
        fn = float(np.sum(pred[y_test == l] != l))
        #print("tp:", tp, " fp:", fp, " fn:", fn,"class:",l,"precision:",tp/(tp+fp),"recall:",tp/(tp+fn))
        if tp > 0:
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
        else:
            (prec, rec) = (0, 1)

        f1.append(2 * prec * rec / (prec + rec))
	recall.append(rec)
        true_predict.append(tp)

    # compute total accuracy
    true_predict.append(float(np.sum(pred[y_test == 2] == 2))) #neutral class to calculate accuracy

    tacc = np.sum(true_predict) / y_test.size

    f1 = np.average(f1)
	
    recall=np.average(recall)

    print true_predict
    print("f1 = %0.3f" % f1)
    print("tacc = %0.3f" % tacc)
    print("rec = %0.3f" % recall)
    return (tacc, f1)



def load_a_dynamic(test_split=0.2,sentiment_embeddings=False,Load_Test=False):

    # print "Loading and Preparing Data..."
    p1=prepare_tweets(PATH_A_2016)
    p2=prepare_tweets(PATH_DATA_2013)
    p1.prepare()
    p2.prepare()
    if(Load_Test):
        p3=prepare_tweets(PATH_TEST_DATA_A)
        p3.prepare()
        tweets_test=p3.tweets

    tweets=np.concatenate((p1.tweets,p2.tweets),axis=0);
    tag=np.concatenate((p1.tag,p2.tag),axis=0);

    result=[]

    # print "Loading Word2Vec Models and Preparing Data ..."
    embedding_size=200
    model_sg = Word2Vec.load(PATH_W2V_SG)

    model_sentiment = pickle.load( open( PATH_SENTIMENT_EMBEDDING, "rb" ) )
    VocabularySize=2
    embeddings_w2v=[]
    embeddings_sentiment=[]
    embeddings_w2v.append(np.array(embedding_size*[0]))
    embeddings_sentiment.append(np.array(embedding_size*[0]))
    embeddings_w2v.append(np.array(embedding_size*[0]))
    embeddings_sentiment.append(np.array(embedding_size*[0]))
    my_hash={}
    result=[]
    for tweet in tweets:
        tokens=tweet.split()
        tokens_indices=[]
        for i in range(0,min(NUM_TOKEN,len(tokens))):
            if(tokens[i] in my_hash):
                tokens_indices.append(my_hash[tokens[i]])
            else:
                try:
                    if(sentiment_embeddings==False):
                        embeddings_w2v.append(model_sg[tokens[i]])
                    else:
                        embeddings_sentiment.append(model_sentiment[tokens[i]])
                    my_hash[tokens[i]]=VocabularySize
                    tokens_indices.append(VocabularySize)
                    VocabularySize+=1
                except:
                    tokens_indices.append(0)
                    # if(sentiment_embeddings==False):
                    #     embeddings_w2v.append(embedding_size*[0])
                    # else:
                    #     embeddings_sentiment.append(embedding_size*[0])
                    # my_hash[tokens[i]]=VocabularySize
                    # tokens_indices.append(VocabularySize)
                    # VocabularySize+=1

        while(i<NUM_TOKEN-1):
            tokens_indices.append(1)
            i+=1
        result.append(np.array(tokens_indices))

    embeddings_w2v = np.array(embeddings_w2v)
    embeddings_sentiment=np.array(embeddings_sentiment)
    X=np.array(result)

    result=[]
    if(Load_Test):
        for tweet in tweets_test:
            tokens=tweet.split()
            tokens_indices=[]
            for i in range(0,min(NUM_TOKEN,len(tokens))):
                if(tokens[i] in my_hash):
                    tokens_indices.append(my_hash[tokens[i]])
                else:
                    tokens_indices.append(0)

            while(i<NUM_TOKEN-1):
                tokens_indices.append(1)
                i+=1
            result.append(np.array(tokens_indices))
        X_train=X
        y_train=tag

        X_test=np.array(result)
        y_test=None

        y_train= np_utils.to_categorical(y_train, 3)

    else:
        np.random.seed(SEED)
        np.random.shuffle(X)
        np.random.seed(SEED)
        np.random.shuffle(tag)


        X_train = X[:int(len(X)*(1-test_split))]
        y_train = tag[:int(len(X)*(1-test_split))]

        X_test = X[int(len(X)*(1-test_split)):]
        y_test = tag[int(len(X)*(1-test_split)):]

        y_train= np_utils.to_categorical(y_train, 3)
        y_test= np_utils.to_categorical(y_test, 3)


        print ("negative test= %d") % np.sum(tag[int(len(X)*(1-test_split)):]=='0')
        print ("positive test= %d") % np.sum(tag[int(len(X)*(1-test_split)):]=='1')
        print ("neutral test= %d") % np.sum(tag[int(len(X)*(1-test_split)):]=='2')

	print ("negative train= %d") % np.sum(tag[:int(len(X)*(1-test_split))]=='0')
        print ("positive train= %d") % np.sum(tag[:int(len(X)*(1-test_split))]=='1')
        print ("neutral train= %d") % np.sum(tag[:int(len(X)*(1-test_split))]=='2')


    if(sentiment_embeddings==False):
        embeddings=embeddings_w2v
    else:
        embeddings= embeddings_sentiment

    return (VocabularySize,embeddings),(X_train,y_train),(X_test,y_test)

    # Build vocabulary

    # word_counts = Counter(itertools.chain(*tweets))
    # print(word_counts)
    # Mapping from index to word
    # vocabulary_inv = [x[0] for x in word_counts]
    # Mapping from word to index
    # vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    # return [vocabulary, vocabulary_inv]



def load_sent140():

    # print "Loading and Preparing Data..."

    lines = codecs.open(PATH_SENT140, 'r', 'utf-8').readlines()
   # lines=lines[0:1000]
    tweets=[]
    tag=[]
    for line in lines:
        line=line.strip()
        items=line.split('\t')
        tweets.append(items[2])
        tag.append(items[1])

    tweets=np.array(tweets)
    tag=np.array(tag)
    # print "Loading Word2Vec Models and Preparing Data ..."
    # model_sg = Word2Vec.load(PATH_W2V_SG)

    VocabularySize=1
    # embeddings=[]
    # embeddings.append(np.array(model_sg.layer1_size*[0]))
    tokens_index_hash={}
    result=[]
    for tweet in tweets:
        tokens=tweet.split()
        tokens_indices=[]
        for i in range(0,min(NUM_TOKEN,len(tokens))):
            if(tokens[i] in tokens_index_hash):
                tokens_indices.append(tokens_index_hash[tokens[i]])
            else:
                # try:
                    # embeddings.append(model_sg[tokens[i]])
                tokens_index_hash[tokens[i]]=VocabularySize
                tokens_indices.append(VocabularySize)
                VocabularySize+=1
                # except:
                #     tokens_indices.append(0)

        while(i<NUM_TOKEN-1):
            tokens_indices.append(0)
            i+=1
        result.append(np.array(tokens_indices))

    # embeddings = np.array(embeddings)
    X=np.array(result)
    result=[]


    # np.random.seed(SEED)
    # np.random.shuffle(X)
    # np.random.seed(SEED)
    # np.random.shuffle(tag)


    X_train = X
    y_train = tag

    y_train= np_utils.to_categorical(y_train, 2)





    return (VocabularySize,tokens_index_hash),(X_train,y_train)

    # Build vocabulary

    # word_counts = Counter(itertools.chain(*tweets))
    # print(word_counts)
    # Mapping from index to word
    # vocabulary_inv = [x[0] for x in word_counts]
    # Mapping from word to index
    # vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    # return [vocabulary, vocabulary_inv]



def load_b_dynamic(test_split=0.2,sentiment_embeddings=False,Load_Test=False):

    # print "Loading and Preparing Data..."
    p1=prepare_tweets(PATH_B_2016,IsBD=True)
    # p2=prepare_tweets(PATH_DATA_2013)
    p1.prepare()
    # p2.prepare()
    if(Load_Test):
        p3=prepare_tweets(PATH_TEST_DATA_B,IsBD=True)
        p3.prepare()
        tweets_test=p3.tweets

    tweets=p1.tweets
    tag=p1.tag
    result=[]

    # print "Loading Word2Vec Models and Preparing Data ..."
    embedding_size=200
    model_sg = Word2Vec.load(PATH_W2V_SG)
    model_sentiment = pickle.load( open( PATH_SENTIMENT_EMBEDDING, "rb" ) )
    VocabularySize=2
    embeddings_w2v=[]
    embeddings_sentiment=[]
    embeddings_w2v.append(np.array(embedding_size*[0]))
    embeddings_sentiment.append(np.array(embedding_size*[0]))
    embeddings_w2v.append(np.array(embedding_size*[0]))
    embeddings_sentiment.append(np.array(embedding_size*[0]))
    my_hash={}
    result=[]
    for tweet in tweets:
        tokens=tweet.split()
        tokens_indices=[]
        for i in range(0,min(NUM_TOKEN,len(tokens))):
            if(tokens[i] in my_hash):
                tokens_indices.append(my_hash[tokens[i]])
            else:
                try:
                    if(sentiment_embeddings==False):
                        embeddings_w2v.append(model_sg[tokens[i]])
                    else:
                        embeddings_sentiment.append(model_sentiment[tokens[i]])
                    my_hash[tokens[i]]=VocabularySize
                    tokens_indices.append(VocabularySize)
                    VocabularySize+=1
                except:
                    tokens_indices.append(0)
                    # if(sentiment_embeddings==False):
                    #     embeddings_w2v.append(embedding_size*[0])
                    # else:
                    #     embeddings_sentiment.append(embedding_size*[0])
                    # my_hash[tokens[i]]=VocabularySize
                    # tokens_indices.append(VocabularySize)
                    # VocabularySize+=1

        while(i<NUM_TOKEN-1):
            tokens_indices.append(1)
            i+=1
        result.append(np.array(tokens_indices))


    embeddings_w2v = np.array(embeddings_w2v)
    embeddings_sentiment=np.array(embeddings_sentiment)
    X=np.array(result)

    result=[]
    if(Load_Test):
        for tweet in tweets_test:
            tokens=tweet.split()
            tokens_indices=[]
            for i in range(0,min(NUM_TOKEN,len(tokens))):
                if(tokens[i] in my_hash):
                    tokens_indices.append(my_hash[tokens[i]])
                else:
                    tokens_indices.append(0)

            while(i<NUM_TOKEN-1):
                tokens_indices.append(1)
                i+=1
            result.append(np.array(tokens_indices))
        X_train=X
        y_train=tag

        X_test=np.array(result)
        y_test=None

        y_train= np_utils.to_categorical(y_train, 2)

    else:
        np.random.seed(SEED)
        np.random.shuffle(X)
        np.random.seed(SEED)
        np.random.shuffle(tag)


        X_train = X[:int(len(X)*(1-test_split))]
        y_train = tag[:int(len(X)*(1-test_split))]

        X_test = X[int(len(X)*(1-test_split)):]
        y_test = tag[int(len(X)*(1-test_split)):]

        y_train= np_utils.to_categorical(y_train, 2)
        y_test= np_utils.to_categorical(y_test, 2)


        print ("negative test= %d") % np.sum(tag[int(len(X)*(1-test_split)):]=='0')
        print ("positive test= %d") % np.sum(tag[int(len(X)*(1-test_split)):]=='1')
        print ("neutral test= %d") % np.sum(tag[int(len(X)*(1-test_split)):]=='2')

    if(sentiment_embeddings==False):
        embeddings=embeddings_w2v
    else:
        embeddings= embeddings_sentiment

    return (VocabularySize,embeddings),(X_train,y_train),(X_test,y_test)

    # Build vocabulary

    # word_counts = Counter(itertools.chain(*tweets))
    # print(word_counts)
    # Mapping from index to word
    # vocabulary_inv = [x[0] for x in word_counts]
    # Mapping from word to index
    # vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    # return [vocabulary, vocabulary_inv]

# load_b_dynamic()
