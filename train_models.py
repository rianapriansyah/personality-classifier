import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
import re
import pickle
import os
import keras
import fasttext
os.environ['KERAS_BACKEND']='tensorflow'
from keras import optimizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Reshape, Concatenate
from keras.layers import Embedding
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, GlobalMaxPooling1D, Bidirectional, LSTM
from keras.layers import concatenate
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

def build_model():
	EMBEDDING_DIM = 300 # how big is each word vector

	#load dataset
	essays = pickle.load(open("essays.p", "rb"))

	X = essays.prepared_w2v
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(X)
	sequences = tokenizer.texts_to_sequences(X)

	word_index = tokenizer.word_index
	MAX_VOCAB_SIZE = len(word_index)
	print('Number of Unique Tokens',len(word_index))

	#because LSTM is stateful that rely on the end of the sequence, the padding will be at the front of sentences
	data_CNN = pad_sequences(sequences, padding='post')
	data_LSTM = pad_sequences(sequences, padding='pre')

	print('CNN input candidate Shape',data_CNN.shape)
	print('LSTM input candidate Shape',data_CNN.shape)

	
	print('Loading Word Embeddings...')
	w2v = KeyedVectors.load('word2vec/idwiki_word2vec_300.model')
	print('Word2vec loaded')

	#Word2vec feature maps
	embedding_matrix_w2v = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
	for word, i in word_index.items():
		try:
			embedding_vector = w2v.wv.word_vec(word)
		except KeyError:
			embedding_vector = np.random.uniform(EMBEDDING_DIM)*-2 + 1

	embedding_matrix_w2v[i] = embedding_vector


	fastText = fasttext.load_model("fasttext/idwiki_fasttext_300.bin")
	print('fastText loaded')

	#fastText feature maps
	embedding_matrix_fasttext = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
	for word, i in word_index.items():
		embedding_vector = fastText[word]

		embedding_matrix_fasttext[i] = embedding_vector 


if __name__ == "__main__":
	build_model()