# -------------------------------------- IMPORT SECTION ---------------------------------------------#
import re
import pandas as pd
import numpy as np
import ftplib
import io
import keras
import nltk
from os import listdir
from os.path import isfile, join
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing import sequence 
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers.recurrent import LSTM
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
import gensim.downloader as api
from gensim.models import Word2Vec
# ----------------------------------------- METHODS ------------------------------------------------#
def intervalNames(path):
        f = open(path, "r")
        lines = f.readlines()
        
        names = [re.findall(r'"([^"]*)"', l)[0] for l in lines if "name" in l]
        
        f.close()

        return names
def feldolgoz(line, f, next_name, index, lists):
        l = line
        while l and next_name not in l:
            if 'text' in l:
                lists[index].append(re.findall(r'"([^"]*)"', l)[0])
            l = f.readline()
        return l
def findTexts(path, names, lists):
        f = open(path, "r")
        line = f.readline()

        i = 0
        while names[i] != "vege" and i < len(names):
            line = feldolgoz(line, f, names[i+1], i, lists)
            i += 1    

        f.close()
def beolvas(path):
        names_results = intervalNames(path)
        names_results.append("vege")

        lists = list()

        for name in names_results:
            l = list()
            l.append(name)
            lists.append(l)

        findTexts(path, names_results, lists)

        return lists, names_results
def writeToFile(path, text):
        f = open("files/" + path,"w") 

        for i in text:
            for j in i:
                f.write(j + '\n')

        f.close() 
def tablazatba(df, interval_names, lista, name):  

        index = 0
        for i in range(0, len(interval_names)):
            if interval_names[i] != 'vege':
                df.insert(i, interval_names[i], lista[i], True)
            index = i
        df.insert(index, "textGrid", name, False)
        return df
def replace_element(my_list, old_value, new_value):
        for n, i in enumerate(my_list):
            if i == old_value:
                my_list[n] = new_value
        return my_list
def split_senteces_into_words(text):
    return keras.preprocessing.text.text_to_word_sequence(text,
                                               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                               lower=True,
                                               split=" ")        
char_toNum_switcher = {
        "V" : 1,
        "O" : 2,
        "P" : 3,
        "R" : 4,
        "L" : 5,
        "N" : 6,
        "E" : 7,
        "I" : 8
}
num_toChar_switcher = {
         1: "V",
         2: "O",
         3: "P",
         4: "R",
         5: "L",
         6: "N",
         7: "E",
         8: "I"
}

# ------------------------------------- DOWNLOAD TEXTS ----------------------------------------------#
hostname = 'berber.tmit.bme.hu'
username = 'mtuba'

passw = 'BA5qKB'

ftp = ftplib.FTP(hostname)
ftp.login(username, passw)

for d in ftp.nlst():
    if d == '10':
        for f in ftp.nlst(d):
            if f.endswith("425.TextGrid") or f.endswith("447.TextGrid") or f.endswith("448.TextGrid"):
                continue
            if f.endswith(".TextGrid"):
                r = io.BytesIO()
                # info = ""
                # splits = ""
                # print(info)
                # print(splits)
                print(f.split('/')[1])
                ftp.retrbinary('RETR ' + f, r.write)
                info = r.getvalue().decode(encoding="utf-8")            
                splits = info.split('\n')
                writeToFile(f.split('/')[1], [s.split('\n') for s in splits])
                r.close()

# ---------------------------------- MERGE TEXT INTO DATAFRAME  -------------------------------------#

onlyfiles = [f for f in listdir("files/") if isfile(join("files/", f))]

df = pd.DataFrame()

vmibaj = ["0160.TextGrid",
        "0167.TextGrid",
        "0185.TextGrid",
        "0191.TextGrid",
        "0205.TextGrid",
        "0331.TextGrid",
        "0413.TextGrid",
        "0551.TextGrid",
        "0605.TextGrid",
        "0617.TextGrid",
        "0619.TextGrid",
        "0662.TextGrid",
        "0760.TextGrid"
        ]

for i in onlyfiles:
    print(i)
    if i in vmibaj:
        continue
    l, names = beolvas("files/" + i)
    temp_df = pd.DataFrame()
    temp_df = tablazatba(temp_df, names, l, i)
    df = pd.concat([df, temp_df]).reset_index(drop = True)

df.replace(np.NaN, "", inplace=True)

drop_indexes = df[df['erzelem'] == 'U'].index
df.drop(drop_indexes, inplace=True)

drop_indexes = df[df['erzelem'] == '_ANONYMIZED_'].index
df.drop(drop_indexes, inplace=True)

drop_indexes = df[df['erzelem'] == 'erzelem'].index
df.drop(drop_indexes, inplace=True)

#df

df2 = df

# cols = list(df2.columns.values)

df2 = df2[['textGrid',
            "erzelem",
            'diszpecscer',
            'diszpecser',
            'diszpecser 2',
            'diszpecser 3',
            'diszpecser1',
            'diszpecser2',
            'diszpecser3',
            'szerelo',
            'ugyfel',
            'ugyfel1',
            'ugyfel2',
            'uygfeé']]
df2

df = df2

df

# --------------------------------- SPLIT INTO TWO DATAFRAME ----------------------------------------#

ugyfel_df = df[['textGrid',
            "erzelem",
            'ugyfel',
            'ugyfel1',
            'ugyfel2',
            'uygfeé']]

diszpecser_df = df[['textGrid',
            "erzelem",
            'diszpecscer',
            'diszpecser',
            'diszpecser 2',
            'diszpecser 3',
            'diszpecser1',
            'diszpecser2',
            'diszpecser3',
            'szerelo']]

ugyfel_df
diszpecser_df

ugyfel_df["All"] = ugyfel_df[ugyfel_df.columns[2:]].apply(
    lambda x: '/'.join(x.dropna().astype(str)), axis = 1)
ugyfel_df["All"]

diszpecser_df["All"] = diszpecser_df[diszpecser_df.columns[2:]].apply(
    lambda x: '/'.join(x.dropna().astype(str)), axis = 1)
diszpecser_df["All"]

ugyfel_df = ugyfel_df[['textGrid', "erzelem", "All"]]
diszpecser_df = diszpecser_df[['textGrid', "erzelem", "All"]]

ugyfel_df.head(10)
diszpecser_df.head(10)

drop_indexes = ugyfel_df[ugyfel_df['All'] == '///'].index
ugyfel_df.drop(drop_indexes, inplace=True)
ugyfel_df.head(10)

drop_indexes = diszpecser_df[diszpecser_df['All'] == '///////'].index
diszpecser_df.drop(drop_indexes, inplace=True)
diszpecser_df.head(10)

replace_chars = ["/",
                 "(",
                 ")",
                 ","
                ]

for char in replace_chars:
    print(char)
    ugyfel_df['All'] = ugyfel_df['All'].str.replace(char,'')    
    diszpecser_df['All'] = diszpecser_df['All'].str.replace(char,'')

ugyfel_df
diszpecser_df
# --------------------------------------- PROCESS TEXTS -----------------------------------------------#
# ------------------------------- MAKE TWO LIST - TEXT, TARGET ----------------------------------------#
ugyfelList = ugyfel_df['All'].tolist()
ugyfelList
ugyfelList[0]

diszpecserList = diszpecser_df['All'].tolist()
diszpecserList

texts = ugyfelList + diszpecserList
texts
len(texts)

#make target list
ugyfelTargetList = ugyfel_df['erzelem'].tolist()
diszpecserTargetList = diszpecser_df['erzelem'].tolist() 

# ugyfelTargetList
# diszpecserTargetList

# distinct_ugyfel = list(set(ugyfelTargetList))
# distinct_ugyfel
# 
# distinct_diszpecser = list(set(diszpecserTargetList))
# distinct_diszpecser

toDistinct = ['N\t\t\t', 'E ', 'N ', 'NN', ' N', 'N\t', 'N0']
expected = ['N', 'E', 'N', 'N', 'N', 'N', 'N']

for i in range(0, len(toDistinct)):
    diszpecserTargetList = replace_element(diszpecserTargetList, toDistinct[i], expected[i])
    ugyfelTargetList = replace_element(ugyfelTargetList, toDistinct[i], expected[i])

# ugyfelTargetList
# diszpecserTargetList
# 
# distinct_ugyfel = list(set(ugyfelTargetList))
# distinct_ugyfel
# 
# distinct_diszpecser = list(set(diszpecserTargetList))
# distinct_diszpecser

target = ugyfelTargetList + diszpecserTargetList
# target
# len(target)
# 
# len(target) == len(texts)

# ------------------------------------------ FILTER -------------------------------------------------#

# --- All TEXT ---#
#  texts variable
# ---- TARGET ----#
# target variable

# ---- split sentences into words ................................
splittedTexts = [] 

for i in texts:
    splittedTexts.append(split_senteces_into_words(i))

splittedTexts
splittedTexts[0]

# set stopWords
stopWords = set(stopwords.words("Hungarian"))
stopWords 

# ---- filter splitted sentences ................................
filteredTexts = []

for s in splittedTexts:
    filteredTexts.append(splitted_sentence(i))

filteredTexts[0]

def splitted_sentence(sentence):
    splitted_s = [] 
    for w in s:
        if w not in stopWords:
            splitted_s.append(w)
    return splitted_s


# for i in range(0, 15):
#     s = ""
#     f = ""
#     for j in splittedTexts[i]:
#         s += j + " "
#     for k in filteredTexts[i]:
#         f += k + " "
#     print("Splitted: " + s)
#     print("Filtered: " + f)

# ---- remove empty element

allFilteredTexts = []
allTarget = []

for i in range(0, len(filteredTexts)):
    if len(filteredTexts[i]) > 0:
        allFilteredTexts.append(filteredTexts[i])
        allTarget.append(target[i])

# for i in range(0, 15):
#     s = ""
#     f = ""
#     for j in splittedTexts[i]:
#         s += j + " "
#     for k in allFilteredTexts[i]:
#         f += k + " "
#     print("Splitted: " + s)
#     print("Splitted target: " + target[0])
#     print("allFilteredTexts: " + f)
#     print("allFilteredTexts target: " + allTarget[0])

# ------------------------------------------ MAKE WORD2VEC --------------------------------------------#

model = Word2Vec(allFilteredTexts, min_count=1, size= 200, workers=3, window = 5, sg = 1)

allFilteredTexts

model["este"]
model.similarity("este", "éjszaka")
model.similarity("este", "tegnap")
model.similarity("este", "online")

model.most_similar('este')[:5]

def display_closestwords_tsnescatterplot(model, word, size):
    arr = np.empty((0,size), dtype='f')
    word_labels = [word]

    close_words = model.similar_by_word(word)
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)    
    
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)
    
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    plt.scatter(x_coords, y_coords)
    
    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()


from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

allFilteredTexts

display_closestwords_tsnescatterplot(model, 'este', 200) 
display_closestwords_tsnescatterplot(model, 'óra', 200) 
display_closestwords_tsnescatterplot(model, 'online', 200) 
display_closestwords_tsnescatterplot(model, 'hülye', 200) 


similarity = model.wmdistance(allFilteredTexts[0], allFilteredTexts[0])
print("{:.4f}".format(similarity))







# ------------------------------------------- KÁOSZ ---------------------------------------------------#

#---- vocab ---#
processed_inputs = []

for w in eeegybenMind:
    w2 = w.split(" ")
    if len(w2) > 1:
        for www in w2:
            processed_inputs.append(www)
    else:
        processed_inputs.append(w)

processed_inputs

words = sorted(list(set(processed_inputs)))
words
words_to_num = dict((w, i) for i, w in enumerate(words))

input_len = len(processed_inputs)
vocab_len = len(words)
print ("Total number of characters:", input_len)
print ("Total vocab:", vocab_len)

#vocab---

# integer encode the documents
vocab_size = len(words)
encoded_ugyfelList = [one_hot(d, vocab_size) for d in ugyfelList]
encoded_diszpecserList = [one_hot(d, vocab_size) for d in diszpecserList]

encoded_all = []

for w in eeegybenMind:
    #for d in w.split():
    #    print(d)
    encoded_all.append([one_hot(d, vocab_size) for d in w.split()])

encoded_all = [one_hot(d, vocab_size) for d in eeegybenMind]
encoded_all[0]
eeegybenMind[0]

encoded_ugyfelList[0]
print(encoded_ugyfelList)
print(encoded_diszpecserList)

#find max length
max_ugyfel = len(max(ugyfelList, key=len)) 
max_diszpecser = len(max(diszpecserList, key=len))

# set text size
max_textSize = max_ugyfel if max_ugyfel > max_diszpecser else max_diszpecser

ugyfel_X_train = sequence.pad_sequences(encoded_ugyfelList, maxlen = max_textSize)
ugyfel_X_train

diszpecser_X_train = sequence.pad_sequences(encoded_diszpecserList, maxlen = max_textSize)
diszpecser_X_train

t = Tokenizer()
t.fit_on_texts(processed_inputs)
t.word_index

t.word_index["én"]
t.word_index["tegnap"]
t.word_index["este"]

word_to_id = t.word_index
word_to_id = {k:(v+3) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2


id_to_word = {value:key for key,value in word_to_id.items()}
print(' '.join(id_to_word[id] for id in encoded_all[0]))


# print(t.word_counts)
# print(t.document_count)
# print(t.word_index)
# print(t.word_docs)

# ----------------------------------------- ENCODING - TARGET ----------------------------------------#
ugyfelTargetList = ugyfel_df['erzelem'].tolist()

# ugyfelTargetList
# distinct_ugyfel = list(set(ugyfelTargetList))
# distinct_ugyfel

diszpecserTargetList = diszpecser_df['erzelem'].tolist() 
diszpecserTargetList

distinct_diszpecser = list(set(diszpecserTargetList))
distinct_diszpecser

toDistinct = ['N\t\t\t', 'E ', 'N ', 'NN', ' N', 'N\t', 'N0']
expected = ['N', 'E', 'N', 'N', 'N', 'N', 'N']

for i in range(0, len(toDistinct)):
    diszpecserTargetList = replace_element(diszpecserTargetList, toDistinct[i], expected[i])
    ugyfelTargetList = replace_element(ugyfelTargetList, toDistinct[i], expected[i])

# distinct_ugyfel = list(set(ugyfelTargetList))
# distinct_ugyfel

# distinct_diszpecser = list(set(diszpecserTargetList))
# distinct_diszpecser

# print(char_toNum_switcher.get("N"))
# print(num_toChar_switcher.get(6))

encoded_ugyfelTarget = np.asarray([char_toNum_switcher.get(i) for i in ugyfelTargetList])
# encoded_ugyfelTarget

encoded_diszpecserTarget = np.asarray([char_toNum_switcher.get(i) for i in diszpecserTargetList])
# encoded_diszpecserTarget
#---------------------------------------------- CONCAT --------------------------------------------#

all_X = np.concatenate((ugyfel_X_train, diszpecser_X_train), axis=0)   
all_y = np.concatenate((encoded_ugyfelTarget, encoded_diszpecserTarget), axis=0)   

distinct_all_y = list(set(all_y))
distinct_all_y

ugyfel_X_train
diszpecser_X_train
all_X

print("X: Az ügyfél beszédek száma: {}, a diszpécser beszédek száma: {}, a kettő együtt: {}".format(len(ugyfel_X_train), len(diszpecser_X_train), len(all_X)))

encoded_ugyfelTarget
encoded_diszpecserTarget
all_y

print("y: Az ügyfél beszédek száma: {}, a diszpécser beszédek száma: {}, a kettő együtt: {}".format(len(encoded_ugyfelTarget), len(encoded_diszpecserTarget), len(all_y))) 

encoder = LabelEncoder()
encoder.fit(all_y)
encoded_Y = encoder.transform(all_y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
dummy_y
dummy_y[0]


X_train, X_test, y_train, y_test = train_test_split(all_X, dummy_y, test_size=0.2)

# X_train
# X_test
# y_train
# y_test

# Build the model 
embedding_vector_length = 32 
top_words = 5000

model = Sequential() 
model.add(Embedding(top_words, embedding_vector_length, input_length=max_textSize)) 
model.add(LSTM(100)) 
model.add(Dense(8, input_dim=8, activation='relu'))
model.add(Dense(8, activation='softmax')) 
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy']) 
print(model.summary()) 

model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=3, batch_size=64)

scores = model.evaluate(X_test, y_test, verbose=0) 
print("Accuracy: %.2f%%" % (scores[1]*100))

# N = "ő fölvettem a telefont,és egy gépi hang azt közölte, hogy hiány van a számlámon, és ki fogják kapcsolni a telefonomat."
# 
# tmp_padded = sequence.pad_sequences([one_hot(N, vocab_size)], maxlen=max_textSize) 
# array([tmp_padded][0])
# 
# asd = model.predict(array([tmp_padded][0]))
# blabla = [int(round(x)) for x in asd[0]]
# blabla
# 
# P = "mondom először azt se tudtam hogy honna jön a hívás, aztán mikor ((szám))"
# 
# tmp_padded2 = sequence.pad_sequences([one_hot(P, vocab_size)], maxlen=max_textSize)
# 
# array([tmp_padded2][0])
# 
# asd2 = model.predict(array([tmp_padded2][0]))
# blabla2 = [int(round(x)) for x in asd2[0]]
# blabla2
# 
# #reverse lookup
# 
# t.word_index.items()
# 
# word_to_id = t.word_index
# word_to_id = {k:(v+3) for k,v in word_to_id.items()}
# word_to_id["<PAD>"] = 0
# word_to_id["<START>"] = 1
# word_to_id["<UNK>"] = 2
# 
# id_to_word = {value:key for key,value in word_to_id.items()}
# print(' '.join(id_to_word[id] for id in tmp_padded2[0][376:]))