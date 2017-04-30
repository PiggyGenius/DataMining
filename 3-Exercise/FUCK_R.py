#! /usr/bin/env python3
#-*- coding:utf-8 -*-
import itertools
import random
import scipy.sparse as sp
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split


def load_classes():
    classes = []
    class_count = 29
    rows = 0
    vocab_size = 141145
    i = 0
    #  max_size = 0

    with open("BaseReuters-29", "r") as f:
        content = f.readlines()
        rows = len(content)

        docs = sp.lil_matrix((rows, vocab_size))
        for line in content:
            words = line.split(' ')
            words.pop() # remove '\n' at the end
            class_index = (int(words.pop(0)))
            classes.append(class_index)
            for word in words:
                val = word.split(':')
                #  if (int(val[0]) > max_size):
                #      max_size = int(val[0])
                docs[i, int(val[0])] = int(val[1])
            i += 1
    return (rows, vocab_size, classes, class_count, docs)

if __name__=="__main__":
    # Do it only once, the result is stored in classes.npy
    # Then, load it (faster than parsing again)
    #  np.save("classes.npy", load_classes())
    rows, vocab_size, classes, class_count, docs = np.load("classes.npy")

    # We randomly split the dataset using sklearn.train_test_split
    train_size = 52500
    test_size = 18203
    train_values, test_values, train_classes, test_classes = train_test_split(docs, classes, train_size = train_size, test_size = test_size, random_state = 1)

    class_term_frequency = [[[0, 0] for i in range(vocab_size)] for j in range(class_count)]
    class_doc_frequency = [0 for i in range(class_count)]

    cx = train_values.tocoo()
    for i, j, v in zip(cx.row, cx.col, cx.data):
        class_term_frequency[classes[i]-1][j][0] += 1
        class_term_frequency[classes[i]-1][j][1] += v

    print("Size :", (rows, vocab_size))
    print("Number of documents per class :")
    for elt in [[x, classes.count(x)] for x in set(classes)]:
        print("Class", elt[0], ":", elt[1], "documents")
        class_doc_frequency[elt[0]-1] += elt[1]


    # BERNOULLI ###############################################

    # We predict the testing set
    # cx = test_values.tocoo()
    # for i,j,v in zip(cx.row, cx.col, cx.data):
        # print(i,j,v)
        # max_value = 0
        # for k in range(len(classes)):
            # max_value = math.log1p(class_doc_frequency / train_size)


    # MULTINOMIAL #############################################

    # theta_m[k][i] contains the theta of the term t_i in the class k
    theta_m = [[0 for i in range(vocab_size)] for j in range(class_count)]

    for k in range(len(class_term_frequency)):
        # count the total number of words
        total = 0
        for elt in class_term_frequency[k]:
            total += elt[1]

        # compute the frequency of term i divided by the total, for this class k
        no_word_value = 1 / (total + vocab_size)
        for i in range(len(class_term_frequency[k])):
            val = class_term_frequency[k][i][1]
            if val == 0:
                theta_m[k][i] = no_word_value
            else:
                theta_m[k][i] = (class_term_frequency[k][i][1] + 1) / (total + vocab_size)

