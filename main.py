# -*- coding: utf-8 -*-
"""
@author: Farok
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, f1_score
from logReg import SGD, predict_class
from utils import extract_ngrams, get_vocab, vectorise

dataDev = pd.read_csv('data_sentiment/dev.csv', names=['text', 'label'])
dataTest = pd.read_csv('data_sentiment/test.csv', names=['text', 'label'])
dataTrain = pd.read_csv('data_sentiment/train.csv', names=['text', 'label'])

dataDevText = list(dataDev['text'])
dataDevLabel = np.array(dataDev['label'])

dataTestText = list(dataTest['text'])
dataTestLabel = np.array(dataTest['label'])

dataTrainText = list(dataTrain['text'])
dataTrainLabel = np.array(dataTrain['label'])

stop_words = ['a','in','on','at','and','or', 
              'to', 'the', 'of', 'an', 'by', 
              'as', 'is', 'was', 'were', 'been', 'be', 
              'are','for', 'this', 'that', 'these', 'those', 'you', 'i',
             'it', 'he', 'she', 'we', 'they', 'will', 'have', 'has',
              'do', 'did', 'can', 'could', 'who', 'which', 'what', 
             'his', 'her', 'they', 'them', 'from', 'with', 'its']



ngs = extract_ngrams("movie", ngram_range=(2,4), stop_words=stop_words,
               char_ngrams=False)



vocab, df, ngram_counts = get_vocab(dataTrainText,stop_words = stop_words,
                                    keep_topN=2000)



vocabIDToWord = dict(enumerate(vocab))
wordToVocabID = {v: k for k, v in vocabIDToWord.items()}

dataTrainText_ngrams = (extract_ngrams(t, vocab = vocab) for t in dataTrainText)
dataDevText_ngrams = (extract_ngrams(t, vocab = vocab) for t in dataDevText)
dataTestText_ngrams = (extract_ngrams(t, vocab = vocab) for t in dataTestText)


dataTrainTextCount = np.array(vectorise(dataTrainText_ngrams, vocab))
dataTestTextCount = np.array(vectorise(dataTestText_ngrams, vocab))
dataDevTextCount = np.array(vectorise(dataDevText_ngrams, vocab))


wCount, trainingLossCount, devLossCount = SGD(dataTrainTextCount,
                                             dataTrainLabel, dataDevTextCount,
                                             dataDevLabel,
                                             epochs=2000,
                                             lr=0.005,
                                             tolerance=0.0001,
                                             alpha = 0,
                                             print_progress=True)

#Plotting Loss data for Count vectors

pd.DataFrame({'train Loss':trainingLossCount, 'dev Loss':devLossCount}, 
             index=range(1,len(trainingLossCount) + 1)).plot(figsize=(15,10))

plt.grid(True)
plt.show()

#Testing Count vector Scores
preds_te_count = predict_class(dataTestTextCount, wCount)

print('Accuracy:', accuracy_score(dataTestLabel ,preds_te_count))
print('Precision:', precision_score(dataTestLabel ,preds_te_count))
print('F1 Score:', f1_score(dataTestLabel ,preds_te_count))

# Calc TFIDF 

idfs = {k: np.log10(len(dataTrainText)/v) for k,v in df.items()}
def tfidf(X_vec):
    X_tfidf_vec = np.zeros_like(X_vec)
    for i, row in enumerate(X_vec):
        for j, cnt in enumerate(row):
            word = vocabIDToWord[j]
            idf = idfs[word]
            X_tfidf_vec[i, j] = X_vec[i, j] * idf
    return X_tfidf_vec

dataTrainTFIDF = tfidf(dataTrainTextCount)
dataDevTFIDF = tfidf(dataDevTextCount)
dataTestTFIDF = tfidf(dataTestTextCount)

wTFIDF, trainingLossTFIDF, devLossTFIDF = SGD(dataTrainTFIDF,
                                             dataTrainLabel, dataDevTFIDF,
                                             dataDevLabel,
                                             epochs=2000,
                                             lr=0.005,
                                             tolerance=0.0001,
                                             alpha = 0,
                                             print_progress=True)


#Plotting Loss data for TFIDF
pd.DataFrame({'train Loss':trainingLossTFIDF, 'dev Loss':devLossTFIDF}, 
             index=range(1,len(trainingLossTFIDF) + 1)).plot(figsize=(15,10))

plt.grid(True)
plt.show()

#Testing TFIDF Scores
preds_te_count = predict_class(dataTestTFIDF, wTFIDF)

print('Accuracy:', accuracy_score(dataTestLabel ,preds_te_count))
print('Precision:', precision_score(dataTestLabel ,preds_te_count))
print('F1 Score:', f1_score(dataTestLabel ,preds_te_count))

#
