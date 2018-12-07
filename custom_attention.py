import re
from math import sin

import gensim.downloader as api
import numpy as np
import tensorflow as tf
from matplotlib import pyplot
from sklearn.model_selection import train_test_split

path_to_file = "colonel/parsed.txt"

num_examples = 3000
BUFFER_SIZE=200
BATCH_SIZE = 8

tf.enable_eager_execution()


class LangConverter:
    gensim_model = api.load('glove-twitter-25')

    def get_vector(self, word):
        if word == '<pad>':
            return np.full((25), 0., dtype=np.float32)
        if word == '<start>':
            return np.full((25), -1., dtype=np.float32)
        if word == '<end>':
            return np.full((25), 1., dtype=np.float32)
        if word not in self.gensim_model.wv.vocab:
            return np.full((25), 2., dtype=np.float32)
        return self.gensim_model.wv[word].astype(np.float32)

    def get_word(self, vector):
        if np.isclose(vector, np.zeros((25))):
            return '<pad>'
        if np.isclose(vector, np.ones((25))):
            return '<end>'
        if np.isclose(vector, np.full((25), -1)):
            return '<start>'
        if np.isclose(vector, np.full((25), 2)):
            return '<unknown>'
        return self.gensim_model.most_similar(positive=[vector], topn=1)[0]


lang_converter = LangConverter()


def preprocess_sentence(w):
    w = w.lower().strip()
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" ]+', " ", w)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^\w?.!,¿]+", " ", w)
    w = w.rstrip().strip()
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w


def create_dataset(path, num_examples):
    lines = open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]
    return word_pairs


def max_length(tensor):
    return max(len(t) for t in tensor)


def load_dataset(path, num_examples):
    # creating cleaned input, output pairs
    pairs = create_dataset(path, num_examples)

    input_tensor = [[lang_converter.get_vector(s) for s in pair[0].split(' ')] for pair in pairs]
    target_tensor = [[lang_converter.get_vector(s) for s in pair[1].split(' ')] for pair in pairs]

    max_length_inp, max_length_tar = max_length(input_tensor), max_length(target_tensor)

    # Padding the input and output tensor to the maximum length
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor,
                                                                 maxlen=max_length_inp,
                                                                 padding='post',
                                                                 dtype='float32')

    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor,
                                                                  maxlen=max_length_tar,
                                                                  padding='post',
                                                                  dtype='float32')

    return input_tensor, target_tensor, max_length_inp, max_length_tar


input_tensor, target_tensor, max_length_inp, max_length_targ = load_dataset(path_to_file, num_examples)

# Creating training and validation sets using an 80-20 split
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor,
                                                                                                target_tensor,
                                                                                                test_size=0.2)

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(len(input_tensor_train))
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)


def seq_func(x):
    return sin(x)


def fit_lstm(train_x, train_y, nb_epoch, neurons):
    x, y = np.array(train_x), np.array(train_y)
    x = x.reshape(len(x), 1, 1)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(neurons, batch_input_shape=(1, 1, 1), stateful=True))
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(x, y, epochs=1, batch_size=1, verbose=1, shuffle=False)
        model.reset_states()
    return model


series = [seq_func(x) for x in range(36)]
supervised = [seq_func(x + 1) for x in range(len(series))]

lstm_model = fit_lstm(series, supervised, 1000, 16)

train_reshaped = np.array(series[:-1]).reshape(len(series) - 1, 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)

predictions = []
expectations = []
forecast_result = series[-1]
for i in range(36):
    forecast_result_reshaped = np.array([forecast_result]).reshape(1, 1, 1)
    forecast_result = lstm_model.predict(forecast_result_reshaped)[0, 0]
    predictions.append(forecast_result)
    expected = seq_func(len(series) + i)
    expectations.append(expected)
    print('x=%d, Predicted=%f, Expected=%f' % (len(series) + i, forecast_result, expected))

pyplot.plot(series + expectations)
pyplot.plot(series + predictions)
pyplot.show()
