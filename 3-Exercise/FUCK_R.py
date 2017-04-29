#! /usr/bin/env python3
#-*- coding:utf-8 -*-
import scipy.sparse as sp
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_classes():
    classes = []
    rows = 0
    vocab_size = 141145
    i = 0
    #  max_size = 0
    for line in content:
        words = line.split(' ')
        words.pop() # remove '\n' at the end
        classes.append(int(words.pop(0)))
        for word in words:
            val = word.split(':')
            #  if (int(val[0]) > max_size):
            #      max_size = int(val[0])
            docs[i, int(val[0])] = int(val[1])
        i += 1
return (rows, vocab_size, classes, docs)

if __name__=="__main__":
    # Do it only once, the result is stored in classes.npy
    # Then, load it (faster than parsing again)
    #  np.save("classes.npy", load_classes())
    rows, vocab_size, classes, docs = np.load("classes.npy")


    print("Size :", (rows, vocab_size))
    print("Number of documents per class :")
    for elt in [[x, classes.count(x)] for x in set(classes)]:
        print("Class", elt[0], ":", elt[1], "documents")

    # We randomly split the dataset using sklearn.train_test_split
    train_values, test_values, train_classes, test_classes = train_test_split(docs, classes, train_size = 52500, test_size = 18203, random_state = 1)
