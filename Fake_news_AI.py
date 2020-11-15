
#start imports
pip install --upgrade tensorflow-gpu==2.0

pip install plotly
pip install --upgrade nbformat
pip install nltk
pip install spacy # spaCy is an open-source software library for advanced natural language processing
pip install WordCloud
pip install gensim # Gensim is an open-source library for unsupervised topic modeling and natural language processing
import nltk
nltk.download('punkt')

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import nltk
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
# import keras
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Conv1D, MaxPool1D, Bidirectional
from tensorflow.keras.models import Model
from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False) 
import plotly.express as px
from nltk import word_tokenize
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
#end import section

#start preping and importing data

#csv file with fake and real news -training
df_true = pd.read_csv("True.csv")
df_fake = pd.read_csv("Fake.csv")

#check for data frame information of our sample
 #df_true.info()
 #df_fake.info()
#check for null values
 #df_true.isnull().sum()
 #df_fake.isnull().sum()

#adding additional column distinguishing fake/real new with 1 & 0's by assigning 1/0 classes
df_true['isfake'] = 0
df_true.head()

df_fake['isfake'] = 1
df_fake.head()

#using pandas to concatenate both data sets with combined index
df = pd.concat([df_true, df_fake]).reset_index(drop = True)
df

#we eliminate the information related to date
df.drop(columns = ['date'], inplace = True)

#combine title and text
df['original'] = df['title'] + ' ' + df['text']
df.head()

#test
#df['original'][0]

#data cleaning
#eliminate "unnecessary words"

#download unncessary words list
nltk.download("stopwords")
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

#eliminate them from txt + eliminate words that have shorter lenghts than 3 function
#store the separated accepted words in a list
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stop_words:
            result.append(token)
            
    return result
#apply function and store it in a new created column
df['clean'] = df['original'].apply(preprocess)

#test original data:
#df['original'][0]
#test cleaned data (all "long" & "particularly defining" words should appear):
#print(df['clean'][0])

#creating the list I mentioned we needed earlier
list_of_words = []
for i in df.clean:
    for j in i:
        list_of_words.append(j)

#test:
#print(list_of_words)

#get unique (now repeated) words
total_words = len(list(set(list_words)))

#to make sure it worked check:
#print(total_words))
#print(len(list_of_words))

#we join all the words into one string
df['clean_joined'] = df['clean'].apply(lambda x: " ".join(x))

#test:
#df['original'][5]
#df['clean_joined'][5]
#can check with other indexes

#end of preping and importing data

#start visualizing cleaned data

#checkout how many subjects we have an their topics in a bar graph
plt.figure(figsize = (8, 8))
sns.countplot(y = "subject", data = df)

#checkout how many false & fake news we have
plt.figure(figsize = (8, 8))
sns.countplot(y = "isfake", data = df)
#make sure it is relatively balanced

#print words cloud fake news, max 2000 words
plt.figure(figsize = (20,20)) 
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = stop_words).generate(" ".join(df[df.isfake == 1].clean_joined))
plt.imshow(wc, interpolation = 'bilinear')

#print word cloud real news, same instructions as before
plt.figure(figsize = (20,20)) 
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = stop_words).generate(" ".join(df[df.isfake == 0].clean_joined))
plt.imshow(wc, interpolation = 'bilinear')

#find maximum number of words per data set
maxlen = -1
for doc in df.clean_joined:
    tokens = nltk.word_tokenize(doc)
    if(maxlen<len(tokens)):
        maxlen = len(tokens)
#print("The maximum number of words in any document is =", maxlen)

#using plotly.express - interactive plot
fig = px.histogram(x = [len(nltk.word_tokenize(x)) for x in df.clean_joined], nbins = 100)
#test
fig.show()

#end visualization data

#start tokenazation and padding

#obv we can just work with text so we tokenize ie vectorize this txt by turning words into integers sequences

#split data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df.clean_joined, df.isfake, test_size = 0.2)

#we use nltk library to tokenize
tokenizer = Tokenizer(num_words = total_words)
#create vocab index:
tokenizer.fit_on_texts(x_train)
#transform into index vocab:
train_sequences = tokenizer.texts_to_sequences(x_train)
test_sequences = tokenizer.texts_to_sequences(x_test)

#test:
#len(train_sequences)
#test translation done with nltk library:
print("The tokenazation for document\n",df.clean_joined[0],"\n is : ",train_sequences[0])

#padding is defined by wikipedia as: distinct practices which all include adding data to the beginning, middle, or end of a message prior to encryption
#we are going to add padding sequences now:
padded_train = pad_sequences(train_sequences,maxlen = 4405, padding = 'post', truncating = 'post')
padded_test = pad_sequences(test_sequences,maxlen = 4405, truncating = 'post') 
#the inputed maxlen ie 4405 was found previously when finding max len of the tested data, we can substitute this value with any other maxlen were we to analyze other test data or to add data to our analyzed test data

#end tokenazation and padding

#lstm network

#Sequential Model
model = Sequential()

#embeddidng layer:so I can work with my computational power - "reduces" space/amount of data
model.add(Embedding(total_words, output_dim = 130))

#Bi-Directional RNN and LSTM
model.add(Bidirectional(LSTM(130)))

#dense layers
model.add(Dense(130, activation = 'relu'))
#binary data
model.add(Dense(1,activation= 'sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
#test parameters:
model.summary()

y_train = np.asarray(y_train)
#training the model
#10% of data used for cross validation, 90% used for traning - represented in the 0.1
model.fit(padded_train, y_train, batch_size = 64, validation_split = 0.1, epochs = 2)
#we are looking for the error in the training and validation data to go down

#end lstm network

#asses network performance

#if the predicted value is >0.5 it's real,else fake
prediction = []
for i in range(len(pred)):
    if pred[i].item() > 0.5:
        prediction.append(1)
    else:
        prediction.append(0)

#obtain accuracy with sklearn lib
accuracy = accuracy_score(list(y_test), prediction)

print("Model Accuracy : ", accuracy)

#visualizing the confusion matrix with sklearn lib congusion matrix
cm = confusion_matrix(list(y_test), prediction)
plt.figure(figsize = (25, 25))
sns.heatmap(cm, annot = True)

#end asses network performance