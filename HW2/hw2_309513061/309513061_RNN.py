#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import csv
import keras
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, LSTM,Embedding,Bidirectional
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[3]:


train_x = pd.read_csv('news_data/train.csv')['Title'].values
test_x = pd.read_csv('news_data/test.csv')["Title"].values
y = pd.read_csv('news_data/train.csv')["Category"].values
train_y = pd.get_dummies(y).values


# In[4]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_x)
all_encoded_texts = tokenizer.texts_to_sequences(train_x)
all_encoded_texts = np.array(all_encoded_texts)
text = sequence.pad_sequences(all_encoded_texts, maxlen=9)


# In[5]:


all_encoded_texts_test = tokenizer.texts_to_sequences(test_x)
all_encoded_texts_test = np.array(all_encoded_texts_test)
text_test = sequence.pad_sequences(all_encoded_texts_test, maxlen=9)


# In[6]:


word_dict = tokenizer.word_index


# In[7]:


def embed_index(embedded):
    embeddings_index = {}
    f = open(embedded, encoding='utf8')
    for line in f:
        word = line.split()[0]
        coefs = np.asarray(line.split()[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index

def embed_matrix(word_dict,embedding_index,dim=100):
    embedding_matrix = np.zeros((len(word_dict)+1, dim))
    for word, i in word_dict.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


# In[8]:


embedding_index = embed_index('glove.6B.200d.txt')


# In[9]:


embedding_matrix = embed_matrix(word_dict,embedding_index,200)


# In[10]:


embedding_layer = Embedding(3600,200,weights=[embedding_matrix],input_length=9,trainable=False)


# In[11]:


def model_produce(embed_layer,drop_rate,n):
    i = 0
    model = Sequential()
    model.add(embed_layer)
    model.add(Bidirectional(LSTM(200, dropout = drop_rate, recurrent_dropout = 0.3)))
    while i<n:
        model.add(Dense(128,activation='tanh'))
        i+=1
    model.add(Dense(5, activation='softmax'))
    return model


# In[39]:


# model = model_produce(embedding_layer,0.3,5)


# In[40]:


# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# In[41]:


# model.summary()


# In[42]:


# X_train, X_test, y_train, y_test = train_test_split(text, train_y, test_size=0.2, random_state=11)


# In[43]:


# history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test))


# In[16]:


# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])


# In[17]:


class_dict = {0:'business',1:'entertainment',2:'politics',3:'sport',4:'tech'}


# In[44]:


# test_predict = np.argmax(model.predict(text_test),axis=1)


# In[19]:


def result_csv(predict,class_dic):
    pre = []
    for i in predict:
        pre.append(class_dic[i])
    with open('309513061_submission_RNN.csv','w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Id','Category'])
        for j in range(len(pre)):
            writer.writerow([j,pre[j]])


# In[45]:


# result_csv(test_predict,class_dict)


# In[46]:


# model.save_weights('rnn.h5')


# In[47]:


model2 = model_produce(embedding_layer,0.3,5)


# In[48]:


model2.load_weights('rnn.h5')


# In[49]:


model2.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# In[50]:


test_predict = np.argmax(model2.predict(text_test),axis=1)


# In[51]:


result_csv(test_predict,class_dict)


# In[ ]:




