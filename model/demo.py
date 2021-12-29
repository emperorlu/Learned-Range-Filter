import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from sklearn.model_selection import train_test_split


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
data1 = np.array(data)

a = 10000

tbin = bin(a)[2:]
print("1 tbin:",tbin)
print("length",len(tbin))
print("length",type(tbin))

train_texts = [bin(x)[2:]  for x in np.arange(1,a+1)]
#data1[:,0]
print("1 train_texts:",train_texts[:3])
print("length",len(train_texts))
print("length",type(train_texts))

y_train = [x  for x in np.random.randint(0,2,a)]
#data1[:,1]
print("2 y_train:",y_train[:3])
print("length",len(y_train))
print("length",type(y_train))

# train_texts = [s.lower() for s in train_texts]
# print("3 train_texts:",train_texts[:3])
# print("length",len(train_texts))
# print("length",type(train_texts))

tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK') 
tk.fit_on_texts(train_texts)


# alphabet = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
alphabet = "0123456789"
char_dict = {}
for i, char in enumerate(alphabet):
    char_dict[char] = i

tk.word_index = char_dict.copy()
tk.word_index[tk.oov_token] = max(char_dict.values()) + 1
print("4 word_index:",tk.word_index)
print("length",type(tk.word_index))

train_texts = tk.texts_to_sequences(train_texts)
print("5 train_texts:",train_texts[:3])
print("length",len(train_texts))
print("length",type(train_texts))

train_data = pad_sequences(train_texts, maxlen=14, padding='post')


train_data = np.array(train_data, dtype='float32')
print("6 train_data:",train_data[:3])
print("length",len(train_data))
print("length",type(train_data))