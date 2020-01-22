import pandas as pd
import numpy as np
import os
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from spellchecker import SpellChecker
from spellchecker import SpellChecker
from nltk.stem import PorterStemmer

#import the data into pandas
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv("sample_submission.csv")


def remove_numbers(word):
	'''
	The function removes any series of number larger than 2
	Input: A word in a string
	Outup: The string removed of words
	'''
    word = re.sub('[0-9]{5,}', '', word)
    word = re.sub('[0-9]{4}', '', word)
    word = re.sub('[0-9]{3}', '', word)
    word = re.sub('[0-9]{2}', '', word)
    return word

def remove_url(text):
	'''
	Clears a peice of text of any URLs
	Input: String containing text
	'''
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

def remove_punctuation(text,punctuation):
	'''
	Clears text of punctutation
	Inputs: Punctuation in a list that should be cleared and text to be cleared in a string
	'''
	for punc in punctuation:
		text = text.replace(punc,'')
	return text


ps = PorterStemmer()
spell = SpellChecker()
stopwords = list(stopwords.words('english'))
def spell_correction_and_stopword_and_stemming_text(text):
	'''
	input a block of text in a string, the function will split it into words, spellcheck those words and then rejoin them into a peice of text
	'''

	split_text = spell.split_words(text)
	corrected = []
	for word in split_text:
		word = spell.correction(word)
		if word not in stopwords:
			word = ps.stem(word)
			corrected.append(word)

	#corrected = [spell.correction(word) for word in split]
	corrected = " ".join(corrected)
	return corrected

def remove_html(word):
	return BeautifulSoup(text, "lxml").text


#clean the input panda files and return them into the 
punctuation = [".",",",";",":","'",'"',"?","/","-","+","&","(",")","[","]","|"]
def clean_data(df, col):
	'''
	Aplies the data cleaning functions created above to the tweets in column, col, and returns the text into them panda cleaned
	'''
        df[col] = df[col].apply(lambda x: remove_numbers(x))
        print(1)
        df[col] = df[col].apply(lambda x: remove_url(x))
        print(2)
        df[col] = df[col].apply(lambda text: remove_punctuation(text, punctuation))
        print(3)
        df[col] = df[col].apply(lambda x: x.lower())
        print(4)
        df[col] = df[col].apply(lambda x: spell_correction_and_stopword_and_stemming_text(x))
        print(5)
        print('data clean')
        return df

train = clean_data(train , 'text')
test = clean_data(test, 'text')





#Move the tweets from the pandas into a tonkenized form in a string
def tokenize_text(df, col):
	tokenized_headlines = []
	for item in df["text"]:										#breaks down the headline column into individual data points in a list
		tokenized_headlines.append(item.split())
	return tokenized_headlines

y_train = train['target']
train_headlines = tokenize_text(train, 'text')
test_headlines = tokenize_text(test, 'text')









#reduce the number of words included in our model through only considering words that have been  used more than once

unique_tokens = []																		#appears more than once
single_tokens = []																		#appears once

for text in train_headlines:															#select headlines
	for word in text:																#select words in headline
		if word not in single_tokens:													#if word is not in appear once section, put it in there
			single_tokens.append(word)
		elif word in single_tokens and word not in unique_tokens:						#else put in appear more than once
			unique_tokens.append(word)




def fill_bag(text_list):
	number_of_tweets_training = len(text_list)
	bag = pd.DataFrame(0, index=np.arange(number_of_tweets_training), columns=unique_tokens)	

	#now lets fill this panda df
	for i, text in enumerate(text_list):	#i gives range, item gives worlds in headline number i
		for word in text:	#goes through words in headline
			if word in unique_tokens:	#if a word is used more than once
				bag.iloc[i][word] += 1		#count that word
	return bag


train_bag = fill_bag(train_headlines)
test_bag = fill_bag(test_headlines)



word_counts = train_bag.sum(axis=0)
train_bag = train_bag.loc[:,(word_counts >= 5) & (word_counts <= 1000)]
test_bag = test_bag.loc[:,(word_counts >= 5) & (word_counts <= 1000)]




#Train a random forest model and then predict the results

from sklearn.ensemble import RandomForestClassifier

#from sklearn.model_selection import KFold, cross_val_score
model = RandomForestClassifier(n_estimators = 100,
                              min_samples_split = 5,
                              min_samples_leaf = 2,
                              max_features = 'auto',
                              max_depth = None,
                              bootstrap =  True)

model.fit(train_bag, y_train)
rf_regression_pred = model.predict(test_bag)



test1 = pd.read_csv('test.csv')
submission = pd.DataFrame()
submission['id'] = test1['id']
submission['target'] = rf_regression_pred
submission.to_csv("submission1.csv", index=False)





