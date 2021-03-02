import pandas as pd
import re
import csv
import argparse
import re
import pickle
import fasttext

charged_words = []
def data_preparation(dataset_path, nrc_path, test_set):
	#Whitespace tokenization
	#Casefolding
	#Cleaning
	#Filtering
	if not test_set:
		dataset = pd.read_csv(dataset_path, sep=';', encoding = 'utf-8')
	else:
		dataset = pd.read_csv(dataset_path, sep=',', encoding = 'utf-8')
	
	global charged_words
	first_row = True

	with open(nrc_path) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		for row in csv_reader:
			if first_row:
				first_row = False
				continue

			if row[2] == "1":
				charged_words.append(row[1])

	cleaned = []
	filtered = []

	if not test_set:
		for i in dataset['Translated'].iteritems():
			sents = []
			for sent in i[1].split('.'):
				#preprocessing
				sent = preprocessing(sent)
				sents.append(sent)

			cleaned.append(sents)
	else:
		for i in dataset['Tweet'].iteritems():
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
	#dataset.drop(columns='Translated')
	#return dataframe with new attribute. 
	#List of cleaned and filtered documents
	return dataset

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

def text_augmentation(essays):
	essays = essays[['Id','prepared_w2v', 'EXT', 'NEU', 'AGR', 'CON', 'OPN']]
	essays = essays.rename(columns={"prepared_w2v": "augmented_essays", 'EXT' : 'ext', 'NEU':'neu', 'AGR':'agr', 'CON':'con', 'OPN':'opn'})

	essays1 = pd.read_csv('datasets/essays_aug1.csv', sep=',', encoding = 'ISO-8859-1')
	essays1 = essays1[['Id','augmented_essays', 'ext', 'neu', 'agr', 'con', 'opn']]

	essays2 = pd.read_csv('datasets/essays_aug2.csv', sep=',', encoding = 'ISO-8859-1')
	essays2 = essays2[['Id','augmented_essays', 'ext', 'neu', 'agr', 'con', 'opn']]

	essays3 = pd.read_csv('datasets/essays_aug3.csv', sep=',', encoding = 'ISO-8859-1')
	essays3 = essays3[['Id','augmented_essays', 'ext', 'neu', 'agr', 'con', 'opn']]

	combined = pd.concat([essays, essays1, essays2, essays3])
	
	return combined

def create_parser():
	parser = argparse.ArgumentParser(description="Input two required files. Essay and NRC ")
	parser.add_argument('essays_path', help='Path to essays. Ex: /path/to/essays.csv')
	parser.add_argument('nrc_path', help='Path to essays. Ex: /path/to/nrc.csv')

	return parser

def main():
	a = "datasets/essays_multi.csv"
	b = "datasets/nrc.csv"
	c = "datasets/testing_dataset.csv"
	print("Preparing Essays dataset")
	#essays = data_preparation(essays_path, nrc_path)
	#train_data = data_preparation(a, b, False)
	test_data = data_preparation(c, b, True)

	#pickle.dump(train_data, open("train_data.p","wb"))
	pickle.dump(test_data, open("test_data.p","wb"))
	print("train & test data dataset prepared")

if __name__ == "__main__":
	main()

