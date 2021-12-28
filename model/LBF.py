from bloom_filter import BloomFilter
import tensorflow as tf 
import tensorflow_hub as hub
import numpy as np
import os
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
        epochs=10,
        verbose=2)


def test_model(tk, model, test_texts):
  test_texts = [s.lower() for s in test_texts]
  test_texts = tk.texts_to_sequences(test_texts)
  test_data = pad_sequences(test_texts, maxlen=1014, padding='post')
  test_data = np.array(test_data, dtype='float32')
  y =  model.predict(test_data)
  ans =[]
  for f in y:
      ans.append(f[1])
  return ans

def Train_Bloom2(bloom,train_features, train_labels ,tau):
    # gen_data()
    X_train=train_features
    y_train=train_labels
    # model.fit(X_train, y_train, epochs=2, batch_size=256,verbose=1)
    preds=test_model(X_train)
    for i in range(len(preds)):
        if preds[i]<tau:
            if y_train[i]==1:
                bloom.add(str(X_train[i]))
    return bloom

def Test_SLBF(model,bloom1,bloom2,data,tau,prediction):
    output1=[]
    for i in range(len(data)):
        #Bloom1
        if str(data[i]) not in bloom1:
            output1.append(0)
            continue
        #Model
        if prediction[i]>tau:
            output1.append(1)
        elif str(data[i]) in bloom2:
            output1.append(1)
        else:
            output1.append(0)
    return np.array(output1)

def Test_NLBF(model,bloom2,data,tau,prediction):
    output1=[]
    for i in range(len(data)):
        #Bloom1
        if prediction[i]>tau:
            output1.append(1)
        elif str(data[i]) in bloom2:
            output1.append(1)
        else:
            output1.append(0)
    return np.array(output1)

def Test_BF(bloom1, test_data):
  y_pred_bloom = []
  for i in test_data:
    if str(i) in bloom1:
      y_pred_bloom.append(1)
    else:
      y_pred_bloom.append(0)
  y_pred_bloom = np.array(y_pred_bloom)
  return y_pred_bloom

# def main():


error_rates = [0.01*i for i in range(1,11)]
tau = 0.9
accuracies = []
for er in tqdm(range(len(error_rates))):
    classifier_data = []
    bloom1 = BloomFilter(max_elements=25000, error_rate=error_rates[er])
    bloom2 = BloomFilter(max_elements=25000, error_rate=error_rates[er])

for data_point in data:
    if data_point[1]==1:
        bloom1.add(data_point[0])
        
for data_point in data:
    if data_point[0] in bloom1:
        classifier_data.append(data_point)

bloom2=Train_Bloom2(bloom2,train_features, train_labels,tau)
test_data = data.copy()
y = np.array([i[1] for i in test_data])
test_data = np.array([test_data[i][0] for i in range(len(test_data))])
prediction = test_model(test_data)
# The model is tested on the entire dataset and its prediction is stored for further use
y_pred_sandwich = Test_SLBF(model,bloom1,bloom2,test_data,tau,prediction)
y_pred_normal= Test_NLBF(model,bloom2,test_data,tau,prediction)
y_pred_bloom = Test_BF(bloom1,test_data)
accuracies.append([accuracy_score(y,y_pred_sandwich),accuracy_score(y, y_pred_normal), accuracy_score(y,y_pred_bloom)])


# test all filters
print(accuracies)
sandwich = []
normal = []
bloom = []
for i in range(len(error_rates)):
    sandwich.append(100*accuracies[i][0])
    normal.append(100*accuracies[i][1])
    bloom.append(100*accuracies[i][2])
t = error_rates
fig = go.Figure()
# Add traces
fig.add_trace(go.Scatter(x=t, y=sandwich,
                    mode='lines',
                    name='sandwich'))
fig.add_trace(go.Scatter(x=t, y=normal,
                    mode='lines',
                    name='regular'))
fig.add_trace(go.Scatter(x=t, y=bloom,
                    mode='lines',
                    name='bloom'))

fig.update_layout(
    title="Comparision of Learned Bloom Filters",
    xaxis_title="Error Rate of Bloom",
    yaxis_title="Accuracy",
    height = 720,
    width = 1280
)
fig.show()



# if __name__ == "__main__":
    # main()