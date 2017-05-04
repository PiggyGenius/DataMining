#! /usr/bin/env python3
#-*- coding:utf-8 -*-
import random
import scipy.sparse as sp
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

vocab_size = 141144
class_count = 29
rows = 70703
train_size = 52500
test_size = 18203

def read_data():
    classes = []
    docs = []

    with open("BaseReuters-29", "r") as f:
        content = f.readlines()
        for line in content:
            dico = {}
            words = line.split(' ')
            words.pop() # remove '\n' at the end
            classes.append(int(words.pop(0))-1)
            for word in words:
                val = word.split(':')
                dico[int(val[0])-1] = int(val[1])
            docs.append(dico)
    return classes, docs


def count_documents(train_classes):
    return [train_classes.count(x) for x in range(class_count)]


def train_multinomial(values, classes):
    m = len(values)
    counts = count_documents(classes)

    D = np.zeros(class_count)
    Pi = np.zeros(class_count)
    tf = np.zeros((class_count, vocab_size))
    PC = np.zeros((class_count, vocab_size))

    for k in range(class_count):
        Pi[k] = counts[k]/m

    for k, doc in zip(classes, values):
        for i in doc:
            tf[k][i] += doc[i]
            D[k] += tf[k][i]

    for k in range(class_count):
        for i in range(vocab_size):
            PC[k][i] = (tf[k][i] + 1)/(D[k] + vocab_size)

    return Pi, PC


def test_multinomial():
    return

def bernoulli_model(train_classes, train_values, test_classes, test_values, vocab_size, train_size):
    # TRAINING
    print("Training model")
    _, N = np.unique(train_classes, return_counts = True)
    pi = N / train_size
    df = np.zeros((vocab_size, class_count))
    for k, document in zip(train_classes, train_values):
        for token in document:
            df[token, k] += 1
    pc = (df + 1) / (N + 2)

    # TESTING
    print("Testing model")
    prediction = np.zeros(test_size)
    for i, document in zip(range(test_size), test_values):
        pc_t = np.copy(pc)
        missing_token = np.array([token not in document for token in range(vocab_size)])
        pc_t[missing_token] *= -1
        pc_t[missing_token] += 1
        pc_t = np.log(pc_t)
        pif = np.sum(pc_t, axis = 0)
        pif += np.log(pi)
        prediction[i] = np.argmax(pif)

    print("Correctly classified samples : %.2f" % accuracy_score(prediction, test_classes))


if __name__=="__main__":
    print("Reading data")
    classes, docs = read_data()

    # We randomly split the dataset using sklearn.train_test_split
    print("Splitting dataset")
    train_values, test_values, train_classes, test_classes = train_test_split(docs, classes, train_size = train_size, test_size = test_size, random_state = 1)

    bernoulli_model(train_classes, train_values, test_classes, test_values, vocab_size, train_size)
