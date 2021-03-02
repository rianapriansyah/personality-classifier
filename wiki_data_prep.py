import argparse
import re
import os
import gensim
import time
import fasttext
from datetime import timedelta
import multiprocessing
from gensim.models import word2vec
import numpy as np

def generate_corpus_as_text(path_name, wiki_path):
	start_time = time.time()

	if not path_name.endswith('xml.bz2'):
		print("File is not suppported")
		return

	print('Creating corpus...')

	article_count = 0
	wiki_idn = gensim.corpora.WikiCorpus(path_name, lemmatize=False, dictionary={})

	with open(wiki_path, 'w', encoding='utf-8') as wiki_txt:
		for text in wiki_idn.get_texts():
			wiki_txt.write(" ".join(text) + os.linesep)
			article_count += 1

			if article_count % 10000 == 0:
				print('{} articles processed'.format(article_count))

		print('total: {} articles'.format(article_count))

	finish_time = time.time()
	print('Corpus created elapsed time: {}'.format(timedelta(seconds=finish_time-start_time)))
	train_word2vec(wiki_path)

def train_word2vec(wiki_path):
	start_time = time.time()
	print('Training Word2Vec Model....')
	sentences = word2vec.LineSentence(wiki_path)
	id_w2v = word2vec.Word2Vec(sentences, size=300, workers=multiprocessing.cpu_count()-1, sg=1)
	id_w2v.save('/Users/rianapriansyah/Workspace/WordEmbeddings/W2V/idwiki_word2vec_200.model')
	finish_time = time.time()

	print('Finished. Elapsed time: {}'.format(timedelta(seconds=finish_time-start_time)))
	print("Word2vec trained")

def train_fasttext(wiki_path):
	start_time = time.time()
	print('Training fastText Model....')
	model = fasttext.train_unsupervised(wiki_path, model='skipgram', dim=300)
	model.save('/Users/rianapriansyah/Workspace/personalityrecognition/fasttext/idwiki_fasttext_300.bin')
	finish_time = time.time()

	print('Finished. Elapsed time: {}'.format(timedelta(seconds=finish_time-start_time)))
	print("fastText trained")

def main():
	source_file = "datasets/idwiki-latest-pages-articles.xml.bz2"
	wiki_path = "datasets/wiki_idn.txt"
	if os.path.isfile(source_file):
		generate_corpus_as_text(source_file, wiki_path)
		#print("Run generate corpus...")
	else:
		print("No such file: {}".format(source_file))
		
if __name__ == "__main__":
	main()


