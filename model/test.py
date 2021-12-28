from bloom_filter import BloomFilter
import tensorflow as tf 
import tensorflow_hub as hub
import numpy as np
import os
from scipy import optimize
from tqdm.notebook import tqdm
from sklearn.metrics import classification_report,accuracy_score
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, Activation, Flatten, Dense
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.models import Model
from copy import deepcopy
from tensorflow.keras.utils import to_categorical
import plotly.graph_objects as go
import random
import plotly.io as pio
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def gen_data():
    fs = open("../data/test_input.txt", "r",encoding='utf-8')
    X = []
    y = []
    c1 = 0
    c2 = 0
    for i, line in enumerate(fs.readlines()[1:]):
        url = line[:-5]
        label = line[-5:-1]
        url.strip(',')
        url.strip("")
        if ((c1==30000) and (c2==30000)):
          break
        if label==",bad":
          if c1<30000:
            c1+=1
            X.append(url)
            y.append(0)
        else:
          if c2<30000:
            c2+=1
            X.append(url)
            y.append(1)
    data = []
    for i in range(len(X)):
        data.append([X[i], y[i]])
    return data

data = gen_data()
train_features = np.array([str(i[0]) for i in data])
train_labels = np.array([i[1] for i in data])

tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK') 
input_size = 1014

embedding_size = 69
conv_layers = [[256, 7, 3],
            [256, 7, 3],
            [256, 3, -1],
            [256, 3, -1],
            [256, 3, -1],
            [256, 3, 3]]

fully_connected_layers = [64, 4]
num_of_classes = 2
dropout_p = 0.1
optimizer = 'adam'
loss = 'categorical_crossentropy'
embedding_weights = []
data1 = deepcopy(data)
data1 = np.array(data1)
train_texts = data1[:,0]
y_train = data1[:,1]
train_texts = [s.lower() for s in train_texts]
tk.fit_on_texts(train_texts)
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
char_dict = {}
for i, char in enumerate(alphabet):
    char_dict[char] = i + 1

tk.word_index = char_dict.copy()
tk.word_index[tk.oov_token] = max(char_dict.values()) + 1
train_texts = tk.texts_to_sequences(train_texts)

# Padding
train_data = pad_sequences(train_texts, maxlen=1014, padding='post')

# Convert to numpy array
train_data = np.array(train_data, dtype='float32')
# test_data = np.array(test_data, dtype='float32')

# =======================Get classes================
# train_y = train_df[0].values
train_class_list = [x  for x in y_train]

train_classes = to_categorical(train_class_list)
# test_classes = to_categorical(test_class_list)
# tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK') 

# =====================Char CNN=======================
# parameter


# Embedding weights
# (70, 69)
vocab_size = len(tk.word_index)
embedding_weights.append(np.zeros(vocab_size))  # (0, 69)

for char, i in tk.word_index.items():  # from index 1 to 69
    onehot = np.zeros(vocab_size)
    onehot[i-1] = 1
    embedding_weights.append(onehot)

embedding_weights = np.array(embedding_weights)
print('Load')

# Embedding layer Initialization
embedding_layer = Embedding(vocab_size + 1,
                            embedding_size,
                            input_length=input_size,
                            weights=[embedding_weights])
inputs = Input(shape=(input_size,), name='input', dtype='int64')  # shape=(?, 1014)
# Embedding
x = embedding_layer(inputs)
# Conv
# for filter_num, filter_size, pooling_size in conv_layers:
x = Conv1D(64, 4)(x)
x = Activation('relu')(x)
# if pooling_size != -1:
x = MaxPooling1D(pool_size=2)(x)  # Final shape=(None, 34, 256)
x = Flatten()(x)  # (None, 8704)
# Fully connected layers
for dense_size in fully_connected_layers:
    x = Dense(dense_size, activation='relu')(x)  # dense_size == 1024
    x = Dropout(dropout_p)(x)
# Output Layer
predictions = Dense(num_of_classes, activation='softmax')(x)
# Build model
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])  # Adam, categorical_crossentropy
model.summary()

model.fit(train_data, train_classes,
        batch_size=256,
        epochs=1,
        verbose=2)

def test_model(test_texts):
    # print("3 test_texts:",test_texts[:3])
    # print("length",len(test_texts))
    test_texts = tk.texts_to_sequences(test_texts)
    # print("4 test_texts:",test_texts[:3])
    # print("length",len(test_texts))
    data = pad_sequences(test_texts, maxlen=1014, padding='post')
    # print("5 data:",data[:3])
    # print("length",len(data))
    data = np.array(data, dtype='float32')
    # print("6 data:",data[:3])
    # print("length",len(data))
    y =  model.predict(data)
    # print("7 y:",y[:3])
    # print("length",len(y))
    ans =[]
    for f in y:
        ans.append(f[1])
    return ans

min_num = 149
max_num = 159
# test_data[max_num:max_num+1]
# test_data = data.copy()
# random.shuffle(test_data)
# print("1 test_data:",test_data[:3])
# print("length",len(test_data))

name=['url','score']
# test=pd.DataFrame(columns=name,data=test_data)
# test.to_csv('data.csv')
# print("before:",test_data[:3])
# print("length",len(test_data))
t = pd.read_csv('data.csv',names=name)
test_data = t[1:].values.tolist()

print("after:",test_data[min_num:max_num])
print("length",len(test_data))

y = np.array([i[1] for i in test_data])
test_data = np.array([test_data[i][0] for i in range(len(test_data))])
# print("1 test_data:",test_data[:3])
# print("length",len(test_data))
# print("2 test_data:",test_data[1:2])
# print("length",len(test_data[1:2]))
# prediction = test_model(test_data[1:2])
# print("8 prediction:",prediction[:3])
# print("length",len(prediction))


def f(x):
    # return (x-1)
    n = int(x)
    print("FFFF:",x,n)
    print("data:",test_data[n:n+1])
    prediction = test_model(test_data[n:n+1])
    print("y:",prediction[0])
    return -prediction[0]


minimum = optimize.minimize_scalar(f, bounds = (min_num, max_num), method = 'bounded', options={'maxiter': 1000})
max = -f(minimum.x)
print("Query Range: (",min_num," , ",max_num,")")
print("Max Score:",max)
if max > 0.9: print("Exist!")
if max < 0.9: print("Not exist!")
