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

algo = "multinomial"
#  algo = "bernoulli" # very slow

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


# Multinomial model ##################################################################################

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
			D[k] += doc[i]

	for k in range(class_count):
		for i in range(vocab_size):
			PC[k][i] = (tf[k][i] + 1.0)/(D[k] + vocab_size)

	return Pi, PC

def test_multinomial(Pi, PC, values):
	prediction = []
	c = 1
	for doc in values:
		print(c, end="\r")
		k_max = 0.0
		PiF_max = -999999999
		for k in range(class_count):
			PiF = np.log(Pi[k])
			for i in doc:
				PiF += doc[i] * np.log(PC[k][i])
				#  PiF += doc[i] * np.log(PC[i, k])
			if PiF > PiF_max:
				k_max = k
				PiF_max = PiF
		prediction.append(k_max)
		c += 1
	print()
	return prediction


# Bernoulli model ####################################################################################

def bernoulli_model(train_classes, train_values, test_classes, test_values):
	# TRAINING
	_, N = np.unique(train_classes, return_counts = True)
	pi = N / train_size
	df = np.zeros((vocab_size, class_count))
	for k, document in zip(train_classes, train_values):
		for token in document:
			df[token, k] += 1
	pc = (df + 1) / (N + 2)

	# TESTING
	prediction = np.zeros(test_size)
	for i, document in zip(range(test_size), test_values):
		print(i, end='\r')
		pc_t = np.copy(pc)
		missing_token = np.array([token not in document for token in range(vocab_size)])
		pc_t[missing_token] *= -1
		pc_t[missing_token] += 1
		pc_t = np.log(pc_t)
		pif = np.sum(pc_t, axis = 0)
		pif += np.log(pi)
		prediction[i] = np.argmax(pif)
	print()
	return accuracy_score(prediction, test_classes)


def train_bernoulli(values, classes):
	return

def test_bernoulli(Pi, PC, values):
	return



if __name__=="__main__":

	print("Reading data")
	classes, docs = read_data()


	print("Size :", (rows, vocab_size))
	print("Number of documents per class :")
	
	i = 1
	for elt in count_documents(classes):
		print("Class", i, ":", elt, "documents")
		i += 1


	iterations = 20
	accuracies = 0.0

	if algo == "multinomial":
		for i in range(iterations):
			train_values, test_values, train_classes, test_classes = train_test_split(docs, classes, train_size = train_size, test_size = test_size)

			Pi, PC = train_multinomial(train_values, train_classes)
			prediction = test_multinomial(Pi, PC, test_values)

			accuracies += accuracy_score(prediction, test_classes)

		print("--- Multinomial model---\nCorrectly classified samples (average on",  iterations, "iterations):", accuracies/iterations)

	elif algo == "bernoulli":
		for i in range(iterations):
			train_values, test_values, train_classes, test_classes = train_test_split(docs, classes, train_size = train_size, test_size = test_size)
			accuracies += bernoulli_model(train_classes, train_values, test_classes, test_values)
		print("--- Bernoulli model---\nCorrectly classified samples (average on",  iterations, "iterations):", accuracies/iterations)



