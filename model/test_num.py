import numpy as np
import os
# from .session import Session
from scipy import optimize
# from tqdm.notebook import tqdm
# from sklearn.metrics import classification_report,accuracy_score
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, Activation, Flatten, Dense
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.models import Model, load_model
from copy import deepcopy
from tensorflow.keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
# import plotly.graph_objects as go
# import random
# import plotly.io as pio
# import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys

# a = int(sys.argv[1]) 
a = 10000

tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK') 

train_texts  = ['{:014b}'.format(x)  for x in np.arange(1,a+1)]
y_texts   = [x  for x in np.random.randint(0,2,a)]

# sdata = []
# for i in range(len(train_texts)):
#     sdata.append([train_texts[i], y_train[i]])
# name=['url','score']
# name=['score']
# stest=pd.DataFrame(columns=name,data=y_texts)
# stest.to_csv('num_score.csv',encoding='gbk')

t = pd.read_csv('num.csv',names=['score'])
y_train = t[1:].values.tolist()
print("y_train:",y_train[:3])
print("length",len(y_train))
print("type",type(y_train))

tk.fit_on_texts(train_texts)
alphabet = "0123456789"

char_dict = {}
for i, char in enumerate(alphabet):
    char_dict[char] = i

tk.word_index = char_dict.copy()
tk.word_index[tk.oov_token] = max(char_dict.values()) + 1

# test_texts = tk.texts_to_sequences(test_texts)
# data = pad_sequences(test_texts, maxlen=1014, padding='post')
# data = np.array(data, dtype='float32')


train_texts = tk.texts_to_sequences(train_texts)

# Padding
train_data = pad_sequences(train_texts, maxlen=14, padding='post')

# Convert to numpy array
train_data = np.array(train_data, dtype='float32')

# =======================Get classes================
train_class_list = [x  for x in y_train]

train_classes = to_categorical(train_class_list)
# print("1 train_data:",train_data[:3])
# print("length",len(train_data))
# print("type",type(train_data))
# print("2 train_classes:",train_classes[:3])
# print("length",len(train_classes))
# print("type",type(train_classes))

# # =====================Char CNN=======================
# # parameter
# input_size = 14
# embedding_size = 11
# conv_layers = [[1024, 7, 3],
#             [1024, 7, 3],
#             [1024, 3, -1],
#             [1024, 3, -1],
#             [1024, 3, -1],
#             [1024, 3, 3]]

# fully_connected_layers = [64, 4]
# num_of_classes = 2
# dropout_p = 0.1
# optimizer = 'adam'
# loss = 'categorical_crossentropy'
# embedding_weights = []
# # Embedding weights
# # (70, 69)
# vocab_size = len(tk.word_index)
# embedding_weights.append(np.zeros(vocab_size))  # (0, 69)

# for char, i in tk.word_index.items():  # from index 1 to 69
#     onehot = np.zeros(vocab_size)
#     onehot[i-1] = 1
#     embedding_weights.append(onehot)

# embedding_weights = np.array(embedding_weights)

# # Embedding layer Initialization
# embedding_layer = Embedding(vocab_size + 1,
#                             embedding_size,
#                             input_length=input_size,
#                             weights=[embedding_weights])
# inputs = Input(shape=(input_size,), name='input', dtype='int64')  # shape=(?, 1014)
# # Embedding
# x = embedding_layer(inputs)
# # Conv
# # for filter_num, filter_size, pooling_size in conv_layers:
# x = Conv1D(64, 4)(x)
# x = Activation('relu')(x)
# # if pooling_size != -1:
# x = MaxPooling1D(pool_size=2)(x)  # Final shape=(None, 34, 256)
# x = Flatten()(x)  # (None, 8704)
# # Fully connected layers
# for dense_size in fully_connected_layers:
#     x = Dense(dense_size, activation='relu')(x)  # dense_size == 1024
#     x = Dropout(dropout_p)(x)
# # Output Layer
# predictions = Dense(num_of_classes, activation='softmax')(x)
# # Build model

# model = Model(inputs=inputs, outputs=predictions)
# model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])  # Adam, categorical_crossentropy
# model.summary()


# model.fit(train_data, train_classes,
#         batch_size=1024,
#         epochs=10000,
#         verbose=2)
# model.save("num_model")

# my_model = load_model("num_model")


# y   = [x  for x in np.random.randint(0,2,a)]

svmclassifier = svm.SVC(kernel='poly', gamma=0.1, decision_function_shape='ovo', C=0.1, verbose=2, epochs=1000)
svmclassifier.fit(train_data, y_train)
print(svmclassifier.score(train_data, y_train))
rf0 = RandomForestClassifier(oob_score=True, random_state=10, epochs=10000, verbose=2)
rf0.fit(train_data, y_train)
print(rf0.oob_score_)


# def test_model(test_texts):
#     # print("3 test_texts:",test_texts[:3])
#     # print("length",len(test_texts))
#     test_texts = tk.texts_to_sequences(test_texts)
#     data = pad_sequences(test_texts, maxlen=1014, padding='post')
#     data = np.array(data, dtype='float32')
#     y =  my_model.predict(data)
#     # print(y, "VS", y1)
#     # if y.all() != y1.all(): print("Error Model!")
#     ans =[]
#     for f in y:
#         ans.append(f[1])
#     return ans

# min_num = int(sys.argv[1])  
# max_num = int(sys.argv[2])


# name=['url','score']
# t = pd.read_csv('data.csv',names=name)
# test_data = t[1:].values.tolist()

# # print("after:",test_data[min_num:max_num])
# # print("length",len(test_data))

# y = np.array([i[1] for i in test_data])
# test_data = np.array([test_data[i][0] for i in range(len(test_data))])
# # prediction = test_model(test_data)


# def f(x):
#     n = int(x)
#     # # print("FFFF:",x,n)
#     # # print("data:",test_data[n:n+1])
#     # prediction = test_model(test_data[n:n+1])
#     # # print("y:",prediction[0])
#     # return -prediction[0]
#     return -n


# minimum = optimize.minimize_scalar(f, bounds = (min_num, max_num), method = 'bounded', options={'maxiter': 1000})
# max = -f(minimum.x)
# print("Query Range: (",min_num,",",max_num,")")
# print("Max Score:",max)
# if max > 0.9: print("Exist!")
# if max < 0.9: print("Not exist!")
