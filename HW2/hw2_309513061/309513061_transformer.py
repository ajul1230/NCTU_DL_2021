#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import csv
import keras
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense,Embedding
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[2]:


x = pd.read_csv('news_data/train.csv')['Title'].values
y = pd.read_csv('news_data/train.csv')['Category'].values
train_y = pd.get_dummies(y).values


# In[3]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(x)
all_encoded_texts = tokenizer.texts_to_sequences(x)
all_encoded_texts = np.array(all_encoded_texts)
text = sequence.pad_sequences(all_encoded_texts, maxlen=9)


# In[4]:


test_x = pd.read_csv('news_data/test.csv')["Title"].values
all_encoded_texts_test = tokenizer.texts_to_sequences(test_x)
all_encoded_texts_test = np.array(all_encoded_texts_test)
text_test = sequence.pad_sequences(all_encoded_texts_test, maxlen=9)


# In[5]:


word_dict = tokenizer.word_index
len(word_dict)


# In[6]:


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


# In[7]:


embedding_index = embed_index('glove.6B.200d.txt')


# In[8]:


embedding_matrix = embed_matrix(word_dict,embedding_index,200)


# In[9]:


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),])
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(0.2)  
    
    def call(self, inputs, training):
        in1 = self.layernorm(inputs)
        attn_output = self.att(in1, in1)
        attn_output = self.dropout(attn_output, training=training)
        out1 = self.layernorm(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout(ffn_output, training=training)
        return out1 + ffn_output


# In[10]:


def model_produce(embed_dim,num_heads,ff_dim,maxlen,word_dict,embedding_matrix,n):
    i = 0
    inputs = layers.Input(shape=(maxlen,))
    embed = Embedding(len(word_dict) + 1,
             embed_dim,
             weights=[embedding_matrix],
             input_length = maxlen,
             trainable=False)

    x = embed(inputs)
    trans = TransformerBlock(embed_dim, num_heads, ff_dim)
    while i<n:
        x = trans(x)
        i+=1
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(60, activation="tanh")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(20, activation="tanh")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(5, activation="softmax")(x)
    model = keras.Model(inputs=inputs,outputs=out)
    return model


# In[11]:


# model = model_produce(200,12,512,9,word_dict,embedding_matrix,3)


# In[12]:


# model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])


# In[13]:


# X_train, X_test, y_train, y_test = train_test_split(text, train_y, test_size=0.2,random_state=11)


# In[14]:


# history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test))


# In[15]:


# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])


# In[25]:


class_dict = {0:'business',1:'entertainment',2:'politics',3:'sport',4:'tech'}
def result_csv(predict,class_dic):
    pre = []
    for i in predict:
        pre.append(class_dic[i])
    with open('309513061_submission_transformer.csv','w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Id','Category'])
        for j in range(len(pre)):
            writer.writerow([j,pre[j]])


# In[17]:


# test_predict = np.argmax(model.predict(text_test),axis=1)


# In[18]:


# result_csv(test_predict,class_dict)


# In[19]:


# model.save_weights('trans.h5')


# In[20]:


model2 = model_produce(200,12,512,9,word_dict,embedding_matrix,3)


# In[21]:


model2.load_weights('trans.h5')


# In[22]:


model2.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])


# In[23]:


test_predict = np.argmax(model2.predict(text_test),axis=1)


# In[24]:


result_csv(test_predict,class_dict)


# In[ ]:




