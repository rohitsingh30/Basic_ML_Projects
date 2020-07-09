text = 'the man was .... they ... then ... the  ... the '
def generateTable(data,k=4):						# building the trasistion table
	T = {}
	for i in range(len(data)-k):
		X =data[i:i+k]
		Y =data[i+k]
		if T.get(X) is None:
			T[X] = {}
			T[X][Y] =1
		else:
			if T[X].get(Y) is None:
				T[X][Y] =1
			else:
				T[X][Y] +=1
		# print(T)									##uncommen to see how is works
	return T 
def convertFreqtoprob(T):
	for kx in T.keys():							# conterting the frequcies to the probabilites
		s = float(sum(T[kx].values()))
		for k in T[kx].keys():
			T[kx][k] = T[kx][k]/s
	print(T)									##uncommen to see how is works
	return T 
text_path = 'speech.txt'
def load_text(filename):
	with open(filename,encoding='utf8') as f:
		return f.read().lower()
text = load_text(text_path)
# print(text)
###  Train our markov chain
def trainmarkov(text,k=4):
	T = generateTable(text,k)
	T = convertFreqtoprob(T)
	return T
model = trainmarkov(text)

import numpy as np
## generating the text by sampling
def sampletext(ctx,T,k):
	ctx = ctx[-k:]
	if T.get(ctx) is None:
		return ' '
	possible_chars = list(T[ctx].keys())
	possible_values = list(T[ctx].values())

	# print(possible_chars)
	return np.random.choice(possible_chars,p=possible_values)
# print(sampletext('comm',model,4))
# definig the geneating fuction
def generatetext(starting_sent,k=4,maxlen=1000):
	sentences = starting_sent
	ctx = starting_sent[-k:]

	for ix in range(maxlen):
		next_predtiction = sampletext(ctx,model,k)
		sentences += next_predtiction
		ctx = sentences[-k:]
	return sentences
print(generatetext('dear'))