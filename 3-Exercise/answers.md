# Explanation on the chosen programming language

## Choice of language
We started doing this exercise in R because the exam is in this language so it seems smart to spend as much time as possible using R to be more confortable with it.
We also wanted to learn more about R, our other option was Python that we know very well (at least for machine learning / data analysis) and we felt that we would learn more doing the exercise in R.

The code for parsing in R was as simple as it is in Python but we had trouble storing data inside the appropriate sparse Matrix.
We used the sparseMatrix data structure from the Matrix package, but filling that matrix was very long (about 1 second per line, which is far too long for our 70 000 lines dataset). We concluded that this data structure was not appropriate to our problem so we decided to switch to Python.

## Data structure
In Python, we decided to use a lil_matrix from the sciPy package : it's a sparse matrix that can be filled iteratively. But manipulating that matrix resulted to be very slow. We finally stored it with a list of dictionnaries : one dictionnary per file, that contains associations word_index-count. It's a lot faster than the previous solution, and therefore the one we kept.

We also ran some parsing scripts at the beginning in order to find the size of the vocabulary, which is now hard coded in the code : it simplifies our reading data function, as it doesn't have to return a long list of arguments.

## Results

Size : (70703, 141144)
Class 1 : 5894 documents
Class 2 : 1003 documents
Class 3 : 2472 documents
Class 4 : 2207 documents
Class 5 : 6010 documents
Class 6 : 2992 documents
Class 7 : 1586 documents
Class 8 : 1226 documents
Class 9 : 2007 documents
Class 10 : 3982 documents
Class 11 : 7757 documents
Class 12 : 3644 documents
Class 13 : 3405 documents
Class 14 : 2307 documents
Class 15 : 1040 documents
Class 16 : 1460 documents
Class 17 : 1191 documents
Class 18 : 1733 documents
Class 19 : 4745 documents
Class 20 : 1411 documents
Class 21 : 1016 documents
Class 22 : 3018 documents
Class 23 : 1050 documents
Class 24 : 1184 documents
Class 25 : 1624 documents
Class 26 : 1296 documents
Class 27 : 1018 documents
Class 28 : 1049 documents
Class 29 : 1376 documents

### For Bernoulli model
Correctly classified samples (average on 1 iterations): 0.55

This code is slow so we didn't have time to run it 20 times.

### For multinomial model
Correcty classified samples (average on 20 iterations): 0.77

