import re

import gensim.downloader as gensim_downloader
import numpy as np
import pandas as pd
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense

WORD_SIZE = 25
gensim_model = gensim_downloader.load('glove-twitter-25')

path_to_file = "colonel/parsed_short.txt"

with open(path_to_file) as dialogues:
    raw_data = [line.strip().split('\t') for line in dialogues.readlines() if len(line.strip()) > 0]


def get_vector(word):
    if word == '<pad>':
        return np.full(WORD_SIZE, 0., dtype=np.float32)
    if word == '<start>':
        return np.full(WORD_SIZE, -1., dtype=np.float32)
    if word == '<end>':
        return np.full(WORD_SIZE, 1., dtype=np.float32)
    if word not in gensim_model.wv.vocab:
        return np.full(WORD_SIZE, 2., dtype=np.float32)
    return gensim_model.wv[word].astype(np.float32)


def get_word(vector):
    if np.allclose(vector, 0.):
        return '<pad>'
    if np.allclose(vector, 1.):
        return '<end>'
    if np.allclose(vector, -1.):
        return '<start>'
    if np.allclose(vector, 2.):
        return '<unknown>'
    return gensim_model.most_similar(positive=[vector], topn=1)[0][0]


def split_sentence(sentense):
    w = sentense.lower().strip()

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,])", r" \1 ", w)
    w = re.sub(r'[ ]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^\w?.!,]+", " ", w)

    w = w.strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    return ['<start>'] + w.split(' ') + ['<end>']


splitted = [(split_sentence(pair[0]), split_sentence(pair[1])) for pair in raw_data]
df = pd.DataFrame.from_records(splitted, columns=['question', 'answer'])

max_question = df['question'].map(lambda x: len(x)).max()
max_answer = df['answer'].map(lambda x: len(x)).max()


def encode_sentence(x, max_len):
    x.extend(['<pad>'] * (max_len - len(x)))
    return np.array([get_vector(w) for w in x])


question_enc = np.array([x for x in df['question'].map(lambda x: encode_sentence(x, max_question)).values])
answer_enc = np.array([x for x in df['answer'].map(lambda x: encode_sentence(x, max_answer)).values])

print(df)

a = Input(shape=(max_question, WORD_SIZE))
b = Dense(1024)(a)
b = Dense(2048)(b)
b = Dense(4096)(b)
c = Dense(WORD_SIZE)(b)
model = Model(inputs=a, outputs=c)
model.compile(loss='mean_squared_error', optimizer='sgd')

model.fit(question_enc, answer_enc, epochs=10)
results = model.predict(question_enc)


def get_words_from_vecs(x):
    return [get_word(vec) for vec in x]


print("\n".join([" ".join(get_words_from_vecs(x)) for x in results]))
