import pickle
import csv
import re
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from sklearn.metrics import hamming_loss

def main():
	data_preparation()
	load_cnn()
	load_lstm()
	load_input()
	predict()

def data_preparation():
	nrc = "datasets/nrc.csv"
	essays_path = "datasets/testing.csv"
	dataset = pd.read_csv(essays_path, sep=';', encoding = 'utf-8')
	
	global charged_words
	first_row = True

	with open(nrc) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=';')
		for row in csv_reader:
			if first_row:
				first_row = False
				continue

			if row[1] == "1":
				charged_words.append(row[0])

	cleaned = []
	filtered = []

	for i in dataset['Text'].iteritems():
		sents = []
		for sent in i[1].split('.'):
			#preprocessing
			sent = preprocessing(sent)
			sents.append(sent)

		cleaned.append(sents)

	for sts in cleaned:
		filtered_sents = []
		#filtering
		for sent in sts:
			if contains_charged_word(sent):
				filtered_sents.append(sent)

		filtered.append(filtered_sents)

	joined = []
	s = " "
	for i in filtered:
		x = s.join(i)
		joined.append(x)

	dataset['processed_text'] = joined
	
	pickle.dump(dataset, open("datasets/testing.p","wb"))
	print("train & test data dataset prepared")

def preprocessing(sent):
	sent = sent.lower()
	sent = re.sub(r"[,.]", "", sent)
	sent = re.sub(r"[^a-zA-Z ]", "", sent)
	sent = re.sub(r"[\s]+", " ", sent)
	return sent.strip()

def contains_charged_word(sent):
	contains_word = False
	words = sent.split(" ")
	match = [w for w in words if w in charged_words]
	if len(match) is not 0:
		contains_word = True

	return contains_word

def predict():
	f = open("output/output.txt", "w")

	predict_cnn(f)
	predict_lstm(f)

	f.close()
	print("Predict succeed. Output file is stored at /output/output.txt")

def predict_cnn(f):
	#testing cnn
	for i in cnn_models:
		predict = cnn_models[i].predict(X)
		predict = predict.round()
		metrics_result(y, predict, i, 0.5, f)

def predict_lstm(f):
	#testing lstm
	for i in lstm_models:
		predict = lstm_models[i].predict(X)
		predict = predict.round()
		metrics_result(y, predict, i, 0.5, f)

def load_input():
	test_set = pickle.load(open("datasets/testing.p", "rb"))

	global y
	y = []
	for i in test_set.iterrows():
		x = []
		x.append(i[1][2])
		x.append(i[1][3])
		x.append(i[1][4])
		x.append(i[1][5])
		x.append(i[1][6])
		y.append(x)

	y = pd.DataFrame(y)
	data = test_set.processed_text

	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(data)
	sequences = tokenizer.texts_to_sequences(data)

	global X 
	X = pad_sequences(sequences, padding='pre', maxlen=1000)
	y = np.asarray(y.replace(['y','n'], [1, 0]))
	print("dataset loaded")

def hamming_score(y_true, y_pred):
	acc_list = []
	for i in range(y_true.shape[0]):
		set_true = set(np.where(y_true[i])[0])
		set_pred = set(np.where(y_pred[i])[0])
		if len(set_true & set_pred) is 0:
			hamming = 0
		else:
			hamming = (len(set_true & set_pred))/(len(set_true | set_pred))
		acc_list.append(hamming)
	return np.round(np.mean(acc_list),decimals=2)

def alpha_evaluation(y_true, y_pred, alpha):
	acclist = []
	for i in range(y_true.shape[0]):
		set_true = set(np.where(y_true[i])[0])
		set_pred = set(np.where(y_pred[i])[0])
		tmp_a = None
		if len(set_true) == 0 and len(set_pred) == 0:
			tmp_a = 1
		else:
			tmp_a = (len(set_true & set_pred))/(len(set_true | set_pred))
		acclist.append(tmp_a**alpha)
	return np.round(np.mean(acclist), decimals=2)

def precision(y_true, y_pred):
	acclist = []
	for i in range(y_true.shape[0]):
		set_true = set(np.where(y_true[i])[0])
		set_pred = set(np.where(y_pred[i])[0])
		tmp_a = None
		if len(set_pred) == 0:
			tmp_a = 1
		else:
			tmp_a = (len(set_true & set_pred))/(len(set_pred))
		acclist.append(tmp_a)
	return np.round(np.mean(acclist), decimals=2)

def recall(y_true, y_pred):	
	acclist = []
	for i in range(y_true.shape[0]):
		set_true = set(np.where(y_true[i])[0])
		set_pred = set(np.where(y_pred[i])[0])
		tmp_a = None
		if len(set_true) == 0:
			tmp_a = 1
		else:
			tmp_a = (len(set_true & set_pred))/(len(set_true))
		acclist.append(tmp_a)
	return np.round(np.mean(acclist), decimals=2)

def metrics_result(y_test, y_pred, title, alpha, f):
	score = hamming_score(y_test, y_pred)
	alpha_score = alpha_evaluation(y_test, y_pred, alpha)
	prec = precision(y_test, y_pred)
	rec = recall(y_test, y_pred)
	fscore = (2*prec*rec)/(prec+rec)
	fscore = np.round(fscore, decimals=2)
	loss = np.round(hamming_loss(y_test, y_pred), decimals=2)

	
	f.write("\n======== {0} ========\n".format(title))
	f.write("Hamming Score {0}\n".format(score))
	f.write("Alpha Evaluation Score {0}\n".format(alpha_score))
	f.write("Precision {0}\n".format(prec))
	f.write("Recall {0}\n".format(rec))
	f.write("F1 Score {0}\n".format(fscore))
	f.write("Hamming loss {0}\n".format(loss))
	f.write("======== End of {0} ========\n".format(title))

def load_cnn():
	cnn_ft_50 = load_model('models/CNN/CNN_ft_thesis_model_50.h5')
	cnn_ft_200 = load_model('models/CNN/CNN_ft_thesis_model_200.h5')
	cnn_ft_300 = load_model('models/CNN/CNN_ft_thesis_model_300.h5')
	cnn_w2v_50 = load_model('models/CNN/CNN_w2v_thesis_model_50.h5')
	cnn_w2v_200 = load_model('models/CNN/CNN_w2v_thesis_model_200.h5')
	cnn_w2v_300 = load_model('models/CNN/CNN_w2v_thesis_model_300.h5')

	global cnn_models
	cnn_models = {
		"CNN FT 50":cnn_ft_50, 
		"CNN FT 200":cnn_ft_200, 
		"CNN FT 300":cnn_ft_300, 
		"CNN W2V 50":cnn_w2v_50, 
		"CNN W2V 200":cnn_w2v_200, 
		"CNN W2V 300":cnn_w2v_300
		}
	print("CNN models loaded")

def load_lstm():
	lstm_ft_50 = load_model('models/LSTM/LSTM_ft_thesis_model_50.h5')
	lstm_ft_200 = load_model('models/LSTM/LSTM_ft_thesis_model_200.h5')
	lstm_ft_300 = load_model('models/LSTM/LSTM_ft_thesis_model_300.h5')
	lstm_w2v_50 = load_model('models/LSTM/LSTM_w2v_thesis_model_50.h5')
	lstm_w2v_200 = load_model('models/LSTM/LSTM_w2v_thesis_model_200.h5')
	lstm_w2v_300 = load_model('models/LSTM/LSTM_w2v_thesis_model_300.h5')

	global lstm_models
	lstm_models = {
		"LSTM FT 50":lstm_ft_50, 
		"LSTM FT 200":lstm_ft_200, 
		"LSTM FT 300":lstm_ft_300, 
		"LSTM W2V 50":lstm_w2v_50, 
		"LSTM W2V 200":lstm_w2v_200, 
		"LSTM W2V 300":lstm_w2v_300
		}
	print("LSTM models loaded")

if __name__ == "__main__":
	charged_words = []
	main()