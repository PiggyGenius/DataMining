# Explanation on the chosen programming language

## Choice of language
We started doing this exercise in R because the exam is in this language so it seems smart to spend as much time as possible using R to be more confortable with it.
We also wanted to learn more about R, our other option was Python that we know very well (at least for machine learning / data analysis) and we felt that we would learn more doing the exercise in R.

The code for parsing in R was as simple as it is in Python but we had trouble storing data inside the appropriate sparse Matrix.
We used the sparseMatrix data structure from the Matrix package, but filling that matrix was very long (about 1 second per line, which is far too long for our 70 000 lines dataset). We concluded that this data structure was not appropriate to our problem so we decided to switch to Python.

## Data structure
In Python, we decided to use a lil_matrix from the sciPy package : it's a sparse matrix that can be filled iteratively. But manipulating that matrix resulted to be very slow. We finally stored it with a list of dictionnaries : one dictionnary per file, that contains associations word_index-count. It's a lot faster than the previous solution, and therefore the one we kept.

We also ran some parsing scripts at the beginning in order to find the size of the vocabulary, which is now hard coded in the code : it simplifies our reading data function, as it doesn't have to return a long list of arguments.

## Answers
Here is the output of our script : 

### For Bernoulli model
### For multinomial model

