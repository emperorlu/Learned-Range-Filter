import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy import optimize

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

test_data = gen_data()
print("1 test_data:",test_data[:3])
print("length",len(test_data))
test_texts = np.array([test_data[i][0] for i in range(len(test_data))])
print("2 test_texts:",test_texts[:3])
print("length",test_texts.shape())
tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK') 
test_texts = [s.lower() for s in test_texts]
print("3 test_texts:",test_texts[:3])
print("length",test_texts.shape())
test_texts = tk.texts_to_sequences(test_texts)
print("4 test_texts:",test_texts[:3])
print("length",test_texts.shape())
data = pad_sequences(test_texts, maxlen=1014, padding='post')
print("5 data:",test_texts[:3])
print("length",data.shape())
data = np.array(data, dtype='float32')
print("6 data:",test_texts[:3])
print("length",data.shape())
# y =  model.predict(test_data)


def f(x):
    return (x-'1')

minimum = optimize.fminbound(f, '2', '6')
print("minimum:",minimum)
#     test_texts = [s.lower() for s in test_texts]
#     test_texts = tk.texts_to_sequences(test_texts)
#     test_data = pad_sequences(test_texts, maxlen=1014, padding='post')
#     test_data = np.array(test_data, dtype='float32')
#     y =  model.predict(test_data)

#     return y



# minimum = optimize.fminbound(f, 'andysgame', 'ceskarepublika')