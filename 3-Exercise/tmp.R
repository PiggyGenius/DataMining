library(Matrix)
all_data = readLines("BaseReuters-29", n=100)
rows = length(all_data)
vocab_size = 1000000 # 141144 ?

# TODO : justify structure
# STRUCTURE = sparse matrix + list of classes
words = Matrix(rows, vocab_size, sparse=TRUE)
classes = vector(,rows)

i = 1
max = 0

parseLine = function(line) {
	print(i)
	split = strsplit(line, " ")[[1]]
	classes[i] = as.numeric(split[1])
	lapply(split[-1], parseVocab)
	i <<- i+1
}

parseVocab = function(word) {
	cols = c()
	w = c()

	val = as.numeric(strsplit(word, ":")[[1]])
	if (val[1] > max) {
		max <<- val[1]
	}
	cols = c(cols, val[1])
	w = c(w, val[2])

	words[i, cols] = w
}

lapply(all_data, parseLine)

print(max)

