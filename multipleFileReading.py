# -------------------------------------- IMPORT SECTION ---------------------------------------------#
import re
import pandas as pd
import numpy as np
import ftplib
import io
import keras
import nltk
import gensim.downloader as api
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
def encode(text, tokenizer):
    return [tokenizer.word_index[i] for i in text]

def decode(encoded_text, rwm):
    print(' '.join(rwm[id] for id in encoded_text))

char_toNum_switcher = {
        "V" : 0,
        "O" : 1,
        "P" : 2,
        "R" : 3,
        "L" : 4,
        "N" : 5,
        "E" : 6,
        "I" : 7
}
num_toChar_switcher = {
         0: "V",
         1: "O",
         2: "P",
         3: "R",
         4: "L",
         5: "N",
         6: "E",
         7: "I"
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
path = "C:/Users/z003w5tm/Documents/BME/code/ProjectLab_2-master/files/"

onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
onlyfiles

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
    l, names = beolvas(path + i)
    temp_df = pd.DataFrame()
    temp_df = tablazatba(temp_df, names, l, i)
    df = pd.concat([df, temp_df]).reset_index(drop = True)

df

df.replace(np.NaN, "", inplace=True)

drop_indexes = df[df['erzelem'] == 'U'].index
df.drop(drop_indexes, inplace=True)

drop_indexes = df[df['erzelem'] == '_ANONYMIZED_'].index
df.drop(drop_indexes, inplace=True)

drop_indexes = df[df['erzelem'] == 'erzelem'].index
df.drop(drop_indexes, inplace=True)

df

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

f = open("C:/Users/z003w5tm/Documents/BME/code/ProjectLab_2-master/stopWords.txt", "r", encoding="utf-8")
lines = f.readlines()
lines = [i.replace("\n", "") for i in lines]
lines

f.close()

# set stopWords
stopWords = set(stopwords.words("Hungarian"))
stopWords = lines
stopWords


# ---- filter splitted sentences ................................
filteredTexts = []

def splitted_sentence(sentence):
    splitted_s = [] 
    for w in s:
        if w not in stopWords:
            splitted_s.append(w)
    return splitted_s

for s in splittedTexts:
    filteredTexts.append(splitted_sentence(i))

filteredTexts[0]


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

# ----------------------------------------- TOKENIZE TEXT -------------------------------------------------#

allFilteredTexts
allTarget

len(allFilteredTexts) == len(allTarget)

t = Tokenizer()
t.fit_on_texts(texts)

#word map for decióodng a text
reverse_word_map = dict(map(reversed, t.word_index.items()))

# t.word_index["tegnap"]
# t.word_index["este"]
# 
# asd = encode(allFilteredTexts[0], t)
# asd
# 
# decode(asd, reverse_word_map)

encoded_texts = []
encoded_texts = [encode(i, t) for i in allFilteredTexts]

for i in range(0, 10):
    print(allFilteredTexts[i])
    print(encoded_texts[i])
    decode(encoded_texts[i], reverse_word_map)
# -------------------------------------------- VISUALIZE -------------------------------------------------#

import seaborn as sns
sns.set(style="darkgrid")

labels = pd.DataFrame(allTarget,  columns =['Sense']) 
labels

ax = sns.countplot(x="Sense", data=labels)

withoutN = [i for i in allTarget if i != "N"]
withoutN

labels = pd.DataFrame(withoutN, columns =['Sense']) 

ax = sns.countplot(x="Sense", data=labels)

import matplotlib.pyplot as plt

names = ['Client', 'Dispatcher']
values = [len(ugyfelList), len(diszpecserList)]

plt.bar(names, values)


fig, ax = plt.subplots()

bar_x = [1,2,3]
values = [117147, 63698, 53449]
names = ['All', 'Selected', 'Unnecessary']
bar_label = [117147, 63698, 53449]

def autolabel(rects):
    for idx,rect in enumerate(bar_plot):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                bar_label[idx],
                ha='center', va='bottom', rotation=0)


bar_plot = plt.bar(bar_x, values, tick_label=names)
autolabel(bar_plot)
plt.ylim(0, 120000)
plt.show()




# --------------------------------------- REFINE THE DATASET ----------------------------------------------#

atLeastFive_indexes = []
atLeastFive_indexes = [i for i in range(len(allFilteredTexts)) if len(allFilteredTexts[i]) > 4]

min5_texts = []
min5_target = []

for i in atLeastFive_indexes:
    min5_target.append(allTarget[i])
    min5_texts.append(allFilteredTexts[i])

min5_target
min5_texts

# count non N targets
non_NTargets = []
non_NTargets = sum(map(lambda x : x != "N", min5_target))
non_NTargets

# select all N target indexes
allN_X = []
allN_X = [i for i in range(len(min5_texts)) if min5_target[i] == "N"]
allN_X
len(allN_X) == len(min5_target) - non_NTargets

# select random indexes from all index
import random
sampling_indexes = random.choices(allN_X, k=non_NTargets*2)
sampling_indexes

# select non N indexes from all index
non_N_Indexes = [i for i in range(len(min5_target)) if min5_target[i] != "N"]
non_N_Indexes

#merge indexes
indexes = sampling_indexes + non_N_Indexes
indexes

len(non_N_Indexes) + len(sampling_indexes) == len(indexes)

# select elements 
selectedTexts = [min5_texts[i] for i in indexes]
selectedTexts

selectedTargets = [min5_target[i] for i in indexes]
selectedTargets

for i in non_N_Indexes:
    selectedTexts.append(min5_texts[i])
    selectedTargets.append(min5_target[i])

len(selectedTargets) == (len(non_N_Indexes) *2) + len(sampling_indexes)

len(sampling_indexes)
selectedTexts[1737]
selectedTargets[1737]

selectedTexts[1738]
selectedTargets[1738]


# ----------------------------------------- ENCODE TEXTS --------------------------------------------------#

# encode seleted texts
encoded_texts = []
encoded_texts = [encode(i, t) for i in selectedTexts]


#pad sequences 
max_textSize = len(max(encoded_texts, key=len)) 
max_textSize

X = sequence.pad_sequences(encoded_texts, maxlen = max_textSize)
X

# ------------------------------------ ONE HOT ENCODE TARGET ----------------------------------------------#

numtarget = []
numTarget = [char_toNum_switcher.get(i) for i in selectedTargets]
numTarget

# sorted_nums = sorted(list(set(numTarget)))
# sorted_nums
# 
# for i in sorted_nums:
#     print(num_toChar_switcher.get(i))

encoder = LabelEncoder()
encoder.fit(numTarget)
# convert integers to dummy variables (i.e. one hot encoded)
y = np_utils.to_categorical(numTarget)

y[0]
y[1778]

# ---------------------------------------- BUILD THE MODEL ------------------------------------------------#

len(X) == len(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# X_train
# X_test
# y_train
# y_test


for i in range(8):
    print("The count of target " + num_toChar_switcher.get(i) + " is: " + str(sum(map(lambda x: x == num_toChar_switcher.get(i), selectedTargets))))



labels = pd.DataFrame(selectedTargets,  columns =['Sense']) 
labels
ax = sns.countplot(x="Sense", data=labels)

#           V  O   P  R  L   N  E  I
weights = [12, 14, 4, 0, 10, 2, 8, 6]


# Build the model 
embedding_vector_length = 32 
top_words = len(t.word_index) + 1

model = Sequential() 
model.add(Embedding(top_words, embedding_vector_length, input_length=max_textSize)) 
model.add(LSTM(100)) 
model.add(Dense(8, input_dim=8, activation='relu'))
model.add(Dense(8, activation='softmax')) 
model.compile(loss="categorical_crossentropy" ,optimizer='adam', metrics=['accuracy']) 
print(model.summary()) 

model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=7, batch_size=64, class_weight=weights)

scores = model.evaluate(X_test, y_test, verbose=0) 
print("Accuracy: %.2f%%" % (scores[1]*100))

# ---------------------------------------- TEST THE MODEL ------------------------------------------------#
def model_test(sentence):
    #print(sentence)
    sentence.replace(",", '')
    
    splitted_sentence = split_senteces_into_words(sentence)
    #splitted_sentence
    
    filtered_sentence = []
    for i in splitted_sentence:
        if i not in stopWords:
            filtered_sentence.append(i)
    
    #filtered_sentence
    
    encode_filtered = encode(filtered_sentence, t)
    #encode_filtered
    #decode(encode_filtered, reverse_word_map)
    
    asd = [encode_filtered]
    #asd
    padded_sentence = sequence.pad_sequences(asd, maxlen = max_textSize)
    #padded_sentence
    
    res = model.predict(array([padded_sentence][0]))
    
    #print(res[0])
    for i in range(len(res[0])):
        erzelem = num_toChar_switcher.get(i)
        szazalek = round((res[0][i] * 100), 2)
        print(erzelem + ": " + str(szazalek) + "%")

find = []
for i in range(8):
    find.append(next(x for x, val in enumerate(target) if val == num_toChar_switcher.get(i)))

find

for i in find:
    print("Eredeti szöveg:")
    print(texts[i])
    print("Eredeti érzelem:")
    print(target[i] + "\n\nKiértlkelés:")
    model_test(texts[i])
    print("\n")
    print("--------------------------------------------------")



# confusion matrix 

y_pred = []
y_pred = model.predict(X_test)
y_pred[0]


predicted = []
for i in range(len(y_pred)):
    list_y_pred = y_pred[i].tolist()
    predicted.append(num_toChar_switcher.get(list_y_pred.index(max(list_y_pred))))

predicted

test = []
for i in range(len(y_test)):
    list_test = y_test[i].tolist()
    test.append(num_toChar_switcher.get(list_test.index(max(list_test))))

eredmeny = []

for i in range(len(test)):
    e = 'Test: ' + test[i] + " - Predicted: " + predicted[i]
    eredmeny.append(e)

f = open("eredmenyek.txt", "w") 
 
for i in eredmeny:
    f.write(i + '\n')
 
f.close() 


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test, predicted)
cm

labels = ["V", "O", "P", "R", "L", "N", "E", "I"]

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)

plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)

plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

s = ""
for i in range(8):
    s += num_toChar_switcher.get(i) + "  "

print(s)



ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix') 
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)

cm[0]

pd.DataFrame(cm, index = [i for i in "VOPRLNEI"],
                  columns = [i for i in "VOPRLNEI"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)

######################################################################################################################################################
######################################################################################################################################################

# költségfüggvény szenzitív a ritka osztályra
# ??

######################################################################################################################################################
######################################################################################################################################################

for i in range(8):
    print("The count of " + num_toChar_switcher.get(i) + " is: " + str(sum(map(lambda x: x == num_toChar_switcher.get(i), target))))

# The count of V is: 289
# The count of O is: 22
# The count of P is: 804
# The count of R is: 1
# The count of L is: 137
# The count of N is: 69632
# The count of E is: 1087

# O, R, L, és V össze vonva,  ----> M, mnt merged : 449

atLeastFive_indexes = [i for i in range(len(allFilteredTexts)) if len(allFilteredTexts[i]) > 4]

for i in range(8):
    print("The count of " + num_toChar_switcher.get(i) + " is: " + str(sum(map(lambda x: x == num_toChar_switcher.get(i), allTarget))))

#from selected:
# The count of V is: 216
# The count of O is: 22
# The count of P is: 784
# The count of R is: 1
# The count of L is: 129
# The count of N is: 57302
# The count of E is: 961
# The count of I is: 781

# O, R, L, és V össze vonva,  ----> M, mnt merged : 368
# P : 784
# E : 961
# N : 1000

min5_texts = []
min5_target = []

for i in atLeastFive_indexes:
    min5_target.append(allTarget[i])
    min5_texts.append(allFilteredTexts[i])

min5_target
min5_texts

# count non N targets
non_NTargets = []
non_NTargets = sum(map(lambda x : x != "N", min5_target))
non_NTargets

# select all N target indexes
allN_X = []
allN_X = [i for i in range(len(min5_texts)) if min5_target[i] == "N"]
allN_X
len(allN_X) == len(min5_target) - non_NTargets

# select random indexes from all index
import random
sampling_indexes = random.choices(allN_X, k=non_NTargets*2)
sampling_indexes

# select non N indexes from all index
non_N_Indexes = []
non_N_Indexes = [i for i in range(len(min5_target)) if min5_target[i] not in "N"]
non_N_Indexes

#merge indexes
indexes = sampling_indexes + non_N_Indexes
indexes

len(non_N_Indexes) + len(sampling_indexes) == len(indexes)

# select elements 
selectedTexts = [min5_texts[i] for i in indexes]
selectedTexts

selectedTargets = [min5_target[i] for i in indexes]
selectedTargets

for i in non_N_Indexes:
    selectedTexts.append(min5_texts[i])
    selectedTargets.append(min5_target[i])

len(selectedTargets) == (len(non_N_Indexes) *2) + len(sampling_indexes)

len(sampling_indexes)
selectedTexts[3475]
selectedTargets[3475]

selectedTexts[3476]
selectedTargets[3476]


# O, R, L, és V to M
for i in range(len(selectedTargets)):
    if selectedTargets[i] in "OVRL":
        selectedTargets[i] = "M"

selectedTargets[1778:]

# ----------------------------------------- NEW SWITCHERS --------------------------------------------------#

char_toNum_switcher_2 = {
        "P" : 0,
        "N" : 1,
        "E" : 2,
        "I" : 3,
        "M" : 4
}

num_toChar_switcher_2 = {
         0: "P",
         1: "N",
         2: "E",
         3: "I",
         4: "M"
}

# ----------------------------------------- ENCODE TEXTS --------------------------------------------------#

for i in range(5):
    print("The count of " + num_toChar_switcher_2.get(i) + " is: " + str(sum(map(lambda x: x == num_toChar_switcher_2.get(i), selectedTargets))))

#          P  N  E  I  M
weights = [5, 1, 9, 7, 15]

# encode seleted texts
encoded_texts = []
encoded_texts = [encode(i, t) for i in selectedTexts]


#pad sequences 
max_textSize = len(max(encoded_texts, key=len)) 
max_textSize

X = sequence.pad_sequences(encoded_texts, maxlen = max_textSize)
X

# ------------------------------------ ONE HOT ENCODE TARGET ----------------------------------------------#

numtarget = []
numTarget = [char_toNum_switcher_2.get(i) for i in selectedTargets]
numTarget

# sorted_nums = sorted(list(set(numTarget)))
# sorted_nums
# 
# for i in sorted_nums:
#     print(num_toChar_switcher.get(i))

encoder = LabelEncoder()
encoder.fit(numTarget)
# convert integers to dummy variables (i.e. one hot encoded)
y = np_utils.to_categorical(numTarget)


len(selectedTargets)
selectedTargets[6945:6955]
numTarget[6945:6955]
y[6945:6955]

y[0]

# ---------------------------------------- BUILD THE MODEL ------------------------------------------------#

len(X) == len(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# X_train
# X_test
# y_train
# y_test

# Build the model 
embedding_vector_length = 32 
top_words = len(t.word_index) + 1

model = Sequential() 
model.add(Embedding(top_words, embedding_vector_length, input_length=max_textSize)) 
model.add(LSTM(100)) 
model.add(Dense(5, input_dim=5, activation='relu'))
model.add(Dense(5, activation='softmax')) 
model.compile(loss="categorical_crossentropy" ,optimizer='adam', metrics=['accuracy']) 
print(model.summary()) 

model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=7, batch_size=64, class_weight=weights)

scores = model.evaluate(X_test, y_test, verbose=0) 
print("Accuracy: %.2f%%" % (scores[1]*100))

# ---------------------------------------- TEST THE MODEL ------------------------------------------------#

N = "ő fölvettem a telefont,és egy gépi hang azt közölte, hogy hiány van a számlámon, és ki fogják kapcsolni a telefonomat."
P = "mondom  azt se tudtam hogy honna jön a hívás, aztán mikor ((szám))"
sajat = "nem értem mi olyan nehéz ezen megcsinálni... semmire nem képesek?"
sajat = "nagyon szépen köszönöm a segítségét"
sajat = "Rendben, Köszönöm"

def model_test2(sentence):
    #print(sentence)
    sentence.replace(",", '')
    
    splitted_sentence = split_senteces_into_words(sentence)
    #splitted_sentence
    
    filtered_sentence = []
    for i in splitted_sentence:
        if i not in stopWords:
            filtered_sentence.append(i)
    
    #filtered_sentence
    
    encode_filtered = encode(filtered_sentence, t)
    #encode_filtered
    #decode(encode_filtered, reverse_word_map)
    
    asd = [encode_filtered]
    #asd
    padded_sentence = sequence.pad_sequences(asd, maxlen = max_textSize)
    #padded_sentence
    
    res = model.predict(array([padded_sentence][0]))
    
    #print(res[0])
    for i in range(len(res[0])):
        erzelem = num_toChar_switcher_2.get(i)
        szazalek = round((res[0][i] * 100), 2)
        print(erzelem + ": " + str(szazalek) + "%")


model_test(sajat)

find = []
for i in range(8):
    find.append(next(x for x, val in enumerate(target) if val == num_toChar_switcher.get(i)))

find

for i in find:
    print("Eredeti üzenet:")
    print(texts[i])
    print("Eredeti érzelem:")
    print(target[i] + "\n\nKiértlkelés:")
    model_test2(texts[i])
    print("\n")
    print("--------------------------------------------------")


######################################################################################################################################################
######################################################################################################################################################

atLeastFive_indexes = [i for i in range(len(allFilteredTexts)) if len(allFilteredTexts[i]) > 4]

min5_texts = []
min5_target = []

for i in atLeastFive_indexes:
    min5_target.append(allTarget[i])
    min5_texts.append(allFilteredTexts[i])

min5_target
min5_texts

# count non N targets
non_NTargets = []
non_NTargets = sum(map(lambda x : x != "N", min5_target))
non_NTargets

# select all N target indexes
allN_X = []
allN_X = [i for i in range(len(min5_texts)) if min5_target[i] == "N"]
allN_X
len(allN_X) == len(min5_target) - non_NTargets

# select random indexes from all index
import random
sampling_indexes = random.choices(allN_X, k=non_NTargets*2)
sampling_indexes

# select non N indexes from all index
non_N_Indexes = []
non_N_Indexes = [i for i in range(len(min5_target)) if min5_target[i] != "N"]
non_N_Indexes

#merge indexes
indexes = sampling_indexes + non_N_Indexes
indexes

len(non_N_Indexes) + len(sampling_indexes) == len(indexes)

# select elements 
selectedTexts = [min5_texts[i] for i in indexes]
selectedTexts

selectedTargets = [min5_target[i] for i in indexes]
selectedTargets

for i in non_N_Indexes:
    selectedTexts.append(min5_texts[i])
    selectedTargets.append(min5_target[i])

len(selectedTargets) == (len(non_N_Indexes) *2) + len(sampling_indexes)

len(sampling_indexes)
selectedTexts[3475]
selectedTargets[3475]

selectedTexts[1778]
selectedTargets[1778]


# O, R, L, és V to M
for i in range(len(selectedTargets)):
    if selectedTargets[i] in "OVRLPEI":
        selectedTargets[i] = "M"

selectedTargets

# ----------------------------------------- NEW SWITCHERS --------------------------------------------------#

char_toNum_switcher_3 = {
        "N" : 0,
        "M" : 1
}

num_toChar_switcher_3 = {
         0: "N",
         1: "M"
}

# ----------------------------------------- ENCODE TEXTS --------------------------------------------------#

for i in range(2):
    print("The count of " + num_toChar_switcher_3.get(i) + " is: " + str(sum(map(lambda x: x == num_toChar_switcher_3.get(i), selectedTargets))))


# encode seleted texts
encoded_texts = []
encoded_texts = [encode(i, t) for i in selectedTexts]

#pad sequences 
max_textSize = len(max(encoded_texts, key=len)) 
max_textSize

X = sequence.pad_sequences(encoded_texts, maxlen = max_textSize)
X

# ------------------------------------ ONE HOT ENCODE TARGET ----------------------------------------------#

numtarget = []
numTarget = [char_toNum_switcher_3.get(i) for i in selectedTargets]
numTarget

# sorted_nums = sorted(list(set(numTarget)))
# sorted_nums
# 
# for i in sorted_nums:
#     print(num_toChar_switcher.get(i))

encoder = LabelEncoder()
encoder.fit(numTarget)
# convert integers to dummy variables (i.e. one hot encoded)
y = np_utils.to_categorical(numTarget)


len(selectedTargets)
selectedTargets[1730:1740]
numTarget[1730:1740]
y[1730:1740]

y[0]

# ---------------------------------------- BUILD THE MODEL ------------------------------------------------#

len(X) == len(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# X_train
# X_test
# y_train
# y_test

# Build the model 
embedding_vector_length = 32 
top_words = len(t.word_index) + 1

model = Sequential() 
model.add(Embedding(top_words, embedding_vector_length, input_length=max_textSize)) 
model.add(LSTM(100)) 
model.add(Dense(2, input_dim=2, activation='relu'))
model.add(Dense(2, activation='softmax')) 
model.compile(loss="categorical_crossentropy" ,optimizer='adam', metrics=['accuracy']) 
print(model.summary()) 

model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=5, batch_size=64)

scores = model.evaluate(X_test, y_test, verbose=0) 
print("Accuracy: %.2f%%" % (scores[1]*100))

# ---------------------------------------- TEST THE MODEL ------------------------------------------------#

N = "ő fölvettem a telefont,és egy gépi hang azt közölte, hogy hiány van a számlámon, és ki fogják kapcsolni a telefonomat."
P = "mondom  azt se tudtam hogy honna jön a hívás, aztán mikor ((szám))"
sajat = "nem értem mi olyan nehéz ezen megcsinálni... semmire nem képesek?"
sajat = "nagyon szépen köszönöm a segítségét"
sajat = "Rendben, Köszönöm"

def model_test3(sentence):
    #print(sentence)
    sentence.replace(",", '')
    
    splitted_sentence = split_senteces_into_words(sentence)
    #splitted_sentence
    
    filtered_sentence = []
    for i in splitted_sentence:
        if i not in stopWords:
            filtered_sentence.append(i)
    
    #filtered_sentence
    
    encode_filtered = encode(filtered_sentence, t)
    #encode_filtered
    #decode(encode_filtered, reverse_word_map)
    
    asd = [encode_filtered]
    #asd
    padded_sentence = sequence.pad_sequences(asd, maxlen = max_textSize)
    #padded_sentence
    
    res = model.predict(array([padded_sentence][0]))
    
    #print(res[0])
    for i in range(len(res[0])):
        erzelem = num_toChar_switcher_3.get(i)
        szazalek = round((res[0][i] * 100), 2)
        print(erzelem + ": " + str(szazalek) + "%")


model_test(sajat)

find = []
for i in range(8):
    find.append(next(x for x, val in enumerate(target) if (val == num_toChar_switcher.get(i) and len(texts[x]) > 4)))

find

for i in find:
    print("Eredeti mondat:")
    print(texts[i])
    print("Eredeti érzelem:")
    print(target[i] + "\n\nKiértlkelés:")
    model_test3(texts[i])
    print("\n")
    print("--------------------------------------------------")