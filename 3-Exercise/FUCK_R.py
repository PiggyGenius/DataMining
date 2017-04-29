#! /usr/bin/env python3
#-*- coding:utf-8 -*-

import scipy.sparse as sp
import pandas as pd


classes = []
rows = 0
vocab_size = 1000000

with open("BaseReuters-29", "r") as f:
    content = f.readlines()
    rows = len(content)
    docs = sp.lil_matrix((rows, vocab_size))

    i = 0
    for line in content:
        print(i)
        words = line.split(' ')
        words.pop() # remove '\n' at the end
        classes.append(int(words.pop(0)))
        for word in words:
            val = word.split(':')
            docs[i, int(val[0])] = int(val[1])
        i += 1


print(classes)

