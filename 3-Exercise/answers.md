# Explanation on the chosen programming language

## Choice of language
We started doing this exercise in R because the exam is in this language so it seems smart to spend as much time as possible using R to be more confortable with it.
We also wanted to learn more about R, our other option was Python that we know very well (at least for machine learning / data analysis) and we felt that we would learn more doing the exercise in R.

The parsing in R  was as simple as it is in Python but we had trouble storing data inside the appropriate sparse Matrix.
We used the sparseMatrix data structure from the Matrix package, but filling that matrix was very long (about 1 second per line, which is far too long for our 70 000 lines dataset). We concluded that this data structure was not appropriate to our problem so we decided to switch to Python.

## Data structure
In Python, we use a lil_matrix from the sciPy package : it's a sparse matrix that can be filled iteratively. The reading function needs about 1 minute, but we can then store the results in a file (by uncommenting the line "np.save(...)"). We can then read that file (with the line np.load(...)), that is much faster than parsing the source file again. With the same idea, we also ran some parsing scripts at the beginning in order to find the size of the vocabulary, which is now hard coded in the code : it saved us some time to parse the file because we can create elements at the right size without any pre-parsing.

## Answers
Here is the output of our script : 

