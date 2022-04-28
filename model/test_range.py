import numpy as np
import os
# from .session import Session
from sklearn import model_selection
from scipy import optimize
from sklearn.model_selection import GridSearchCV
# from tqdm.notebook import tqdm
# from sklearn.metrics import classification_report,accuracy_score
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, Activation, Flatten, Dense
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import optimizers
from keras.layers import GRU, LSTM, Bidirectional
from copy import deepcopy
from tensorflow.keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
# import plotly.graph_objects as go
# import random
# import plotly.io as pio
# import matplotlib.pyplot as plt
import sys
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
a = 10000
a1 = a+1
tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK') 

train_texts  = ['{:014b}'.format(x)  for x in range(1,a+1)]
# random.shuffle(train_texts)
adata=np.load('a.npy')
y_train=adata.tolist()

def in_range(x, y):
    for i in range(x, y+1):
        if y_train[i-1] == 1:
            return 1
    return 0

range_data = []
for i in range(3, 21):
    for j in range(1,a1):
        if i+j < a1:
            p = in_range(j, i+j)
            tmp = [j, i+j, p]
            range_data.append(tmp)


name=['min','max','pre']
stest=pd.DataFrame(columns=name,data=range_data)
stest.to_csv('range.csv',encoding='gbk')

# t = pd.read_csv('num_score.csv',names=['score'])
# y_train = t.values.tolist()


tk.fit_on_texts(train_texts)
alphabet = "01"

char_dict = {}
for i, char in enumerate(alphabet):
    char_dict[char] = i + 1

tk.word_index = char_dict.copy()
tk.word_index[tk.oov_token] = max(char_dict.values()) + 1


train_texts = tk.texts_to_sequences(train_texts)

# Padding
train_data = pad_sequences(train_texts, maxlen=14, padding='post')

# Convert to numpy array
train_data = np.array(train_data, dtype='float32')


# 训练阶段
# # =======================Get classes================
# train_class_list = [x  for x in y_train]

# train_classes = to_categorical(train_class_list)
# # print("1 train_data:",train_data[:10])
# # print("length",len(train_data))
# # print("type",type(train_data))
# # print("2 train_classes:",train_classes[:10])
# # print("length",len(train_classes))
# # print("type",type(train_classes))

# # =====================Char CNN=======================
# # parameter
# input_size = 14
# embedding_size = 3
# conv_layers = [[256, 7, 3],
#             # [256, 7, 3],
#             [256, 3, -1],
#             # [256, 3, -1],
#             # [256, 3, -1],
#             [256, 3, 3]]

# fully_connected_layers = [64, 4]
# num_of_classes = 2
# dropout_p = 0.1

# loss = 'categorical_crossentropy'
# embedding_weights = []
# # Embedding weights
# # (70, 69)
# vocab_size = len(tk.word_index)
# print("vocab_size",vocab_size)
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

# # x = GRU(64)(x)
# # x = LSTM(64)(x)
# # Bidirectional(
# x = LSTM(64, activation='relu', return_sequences=True)(x)
# # x = Dropout(0.2)(x)
# x = LSTM(64, activation='relu', return_sequences=False)(x)
# # x = Dropout(0.2)(x)

# # # Conv
# # # for filter_num, filter_size, pooling_size in conv_layers:
# # x = Conv1D(64, 4, kernel_initializer='random_normal')(x)
# # x = Activation('relu')(x)
# # # if pooling_size != -1:
# # x = MaxPooling1D(pool_size=2)(x)  # Final shape=(None, 34, 256)
# # x = Conv1D(64, 4, kernel_initializer='random_normal')(x)
# # x = Activation('relu')(x)
# # # if pooling_size != -1:
# # x = MaxPooling1D(pool_size=2)(x)  # Final shape=(None, 34, 256)

# # x = Flatten()(x)  # (None, 8704)

# # # Fully connected layers
# # for dense_size in fully_connected_layers:
# #     x = Dense(dense_size, activation='relu', kernel_initializer='random_normal')(x)  # dense_size == 1024
# #     x = Dropout(dropout_p)(x)


# # Output Layer
# predictions = Dense(num_of_classes, activation='softmax')(x) #softmax
# # Build model
# # optimizer = optimizers.Adam(learning_rate=0.1, decay=0.001)
# optimizer = 'adam' #'adam' RMSprop
# model = Model(inputs=inputs, outputs=predictions)
# model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])  # Adam, categorical_crossentropy
# model.summary()


# model.fit(train_data, train_classes,
#         batch_size=256,
#         epochs=1000,
#         verbose=1)
# model.save("num_model2")


## 其他模型
# param_grid=[{"kernel":["rbf"],"C":[0.1, 1, 10], "gamma": [1, 0.1, 0.01]},
#             {"kernel":["poly"],"C": [0.1, 1, 10], "gamma": [1, 0.1, 0.01],"degree":[3,5,10],"coef0":[0,0.1,1]},
#             {"kernel":["sigmoid"], "C": [0.1, 1, 10], "gamma": [1, 0.1, 0.01],"coef0":[0,0.1,1]}]

# grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=4, verbose=True)
# grid.fit(train_data,y_train)
# print('grid_best_params:',  grid.best_params_)
# print('grid.best_score_:', grid.best_score_)

# a = int(sys.argv[1]) 
# svmclassifier = svm.SVC(kernel='rbf', gamma=0.1, C=0.9, verbose=1)
# svmclassifier.fit(train_data, y_train)
# print("\nSCV: ",svmclassifier.score(train_data, y_train))
# rf0 = RandomForestClassifier(oob_score=True, random_state=100)
# rf0.fit(train_data, y_train)
# print("RF: ",rf0.oob_score_)










# my_model = load_model("num_model")

## 测试阶段 - filter
# # print("train_data:",train_data[:10])
# # print("length",len(train_data))
# # print("type",type(train_data))

# print("y_train:",y_train[:20])
# # print("length",len(y_train))
# # print("type",type(y_train))


# # data  = ['{:014b}'.format(x)  for x in range(1,2)]

# num = int(sys.argv[1]) 
# data = ['{:014b}'.format(num)]
# data = tk.texts_to_sequences(data)
# # data = pad_sequences(test_texts, maxlen=1014, padding='post')
# data = np.array(data, dtype='float32')
# y =  my_model.predict(data)
# pre = y[0][1]
# # print(y[:20])
# print(y[0])
# if pre > 0.9: print("Exist!", pre)
# if pre < 0.9: print("Not exist!", pre)






"""

## 测试阶段 - Range filter
min_num = int(sys.argv[1]) 
max_num = int(sys.argv[2]) 
def f(x):
    data = ['{:014b}'.format(int(x))]
    data = tk.texts_to_sequences(data)
    data = np.array(data, dtype='float32')
    y = my_model.predict(data)
    prediction = y[0][1]
    return -prediction


minimum = optimize.minimize_scalar(f, bounds = (min_num, max_num), method = 'bounded')#, options={'maxiter': 1000})
max = -f(minimum.x)
print("Query Range: (",min_num,",",max_num,")")
print("Max Score:",max)
if max > 0.9: print("Exist!")
if max < 0.9: print("Not exist!")

for i in range(min_num,max_num):
    max = -f(i)
    if max > 0.9: print(i, ":", max, " Exist!")
    if max < 0.9: print(i, ":", max, " Not Exist!")
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

"""
