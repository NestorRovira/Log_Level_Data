# -*- coding: utf-8 -*-
import yaml
import os
import sys
import re as re
import time
import logging

import multiprocessing
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary

import random as rn
seed_value = 17020
seed_window = 1500

import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional

from src import Helper

# -----------------------------
# Logging setup (ordenado)
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("DeepLV")

# -----------------------------
# TensorFlow (compat 1.x)
# -----------------------------
config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 16})
sess = tf.Session(config=config)
K.set_session(sess)

# -----------------------------
# Config global (sin cambios)
# -----------------------------
csv.field_size_limit(100000000)
sys.setrecursionlimit(1000000)

n_iterations = 1
embedding_iterations = 1
n_epoch = 50

vocab_dim = 100
maxlen = 100
n_exposures = 10
window_size = 7
batch_size = 24
input_length = 100
cpu_count = multiprocessing.cpu_count()

test_list = []
neg_full = []
pos_full = []
syntactic_list = []

# Paths de salida (igual que antes)
model_location = 'model_block' + '/lstm_' + sys.argv[1]
embedding_location = 'embedding_block' + '/Word2vec_model_' + sys.argv[1] + '.pkl'


# -----------------------------
# Utilidades de logging
# -----------------------------
class PhaseTimer:
    def __init__(self, name):
        self.name = name
        self.t0 = None
    def __enter__(self):
        self.t0 = time.time()
        log.info(f"[{self.name}] Inicio")
    def __exit__(self, exc_type, exc_val, exc_tb):
        dt = time.time() - self.t0
        if exc_type is None:
            log.info(f"[{self.name}] Fin en {dt:.2f}s")
        else:
            log.error(f"[{self.name}] Error tras {dt:.2f}s")

def _brief_seq(seq, n=3):
    return seq[:n] + ['...'] + seq[-n:] if len(seq) > 2*n else seq

# -----------------------------
# Carga de datos (igual lógica)
# -----------------------------
def loadfile():
    """
    Carga el CSV generado por block_processing para el dataset indicado
    (por ejemplo: cassandra, flink, kafka...).
    """
    # Dataset viene como argumento del script
    dataset = sys.argv[1].strip()  # "cassandra", "flink", etc.

    # Ruta dinámica al CSV correspondiente
    csv_path_literal = f"block_processing/blocks/logged_syn_{dataset}.csv"

    log.info(f"[Datos] cwd={os.getcwd()}")
    log.info(f"[Datos] CSV esperado: {csv_path_literal}")

    # Comprobar existencia antes de leer (log más claro si falta)
    if not os.path.exists(csv_path_literal):
        log.error(f"[Datos] No se encontró el fichero: {csv_path_literal}")
        raise FileNotFoundError(f"No existe el dataset '{dataset}' o falta su CSV")

    with PhaseTimer(f"Lectura CSV ({dataset})"):
        data_full = pd.read_csv(
            csv_path_literal,
            usecols=['Key', 'Values', 'Level', 'Message'],  # por nombre
            engine='python'
        )

    dataset_values = data_full.values
    classes = dataset_values[:, 2]
    data = data_full['Values'].values.tolist()
    combined = data
    combined_full = data_full.values.tolist()

    log.info(f"[Datos] Filas={len(data_full):,} | Columnas={list(data_full.columns)}")
    log.info(f"[Datos] Ejemplo 'Values': {combined[0] if combined else '(vacío)'}")

    # Codificación y splits igual que antes
    with PhaseTimer("Encode etiquetas"):
        encoder = LabelEncoder()
        encoder.fit(classes)
        encoded_Y = encoder.transform(classes)
        y = Helper.ordinal_encoder(classes)

    with PhaseTimer("Split train/val/test"):
        x_train, x_test, y_train, y_test = train_test_split(
            combined_full, y, test_size=0.2, train_size=0.8,
            random_state=seed_value, stratify=y
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=0.25, train_size=0.75,
            random_state=seed_value, stratify=y_train
        )

    test_block_list, train_block_list = [], []
    for x in x_test:
        test_list.append(x[0])
        test_block_list.append(x[1])
    x_test = np.array(test_block_list)
    for x in x_train:
        train_block_list.append(x[1])
    x_train = train_block_list

    log.info(f"[Datos] Split → train={len(x_train):,} | val={len(x_val):,} | test={len(x_test):,}")

    return combined, y, x_train, x_val, x_test, y_train, y_val, y_test



# -----------------------------
# Tokenización (igual lógica)
# -----------------------------
def word_splitter(word, docText):
    splitted = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', word)).split()
    for word in splitted:
        docText.append(word.lower())

def tokenizer(text):
    newText = []
    for doc in text:
        docText = []
        for word in str(doc).replace("'", "").replace("[", "").replace("]", "").replace(",", "").replace('"', "").split(' '):
            docText.append(word)
        newText.append(docText)
    return newText


# -----------------------------
# Embedding / Diccionarios
# -----------------------------
def input_transform(words):
    model = Word2Vec.load(embedding_location)
    _, _, dictionaries = create_dictionaries(model, words)
    return dictionaries

def create_dictionaries(model=None, combined=None):
    from keras.preprocessing import sequence
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
        w2indx = {v: k + 1 for k, v in gensim_dict.items()}
        w2vec = {word: model.wv[word] for word in w2indx.keys()}

        def parse_dataset(combined):
            data = []
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data

        combined = parse_dataset(combined)
        combined = sequence.pad_sequences(combined, maxlen=maxlen)
        return w2indx, w2vec, combined

def word2vec_train(combined):
    with PhaseTimer("Entrenar Word2Vec"):
        model = Word2Vec(size=vocab_dim,
                         min_count=n_exposures,
                         window=window_size,
                         workers=cpu_count, sg=1,
                         iter=embedding_iterations)
        model.build_vocab(combined)
        model.save(embedding_location)
        index_dict, word_vectors, combined = create_dictionaries(model=model, combined=combined)
    log.info(f"[W2V] Guardado en: {embedding_location}")
    log.info(f"[W2V] Vocab={len(index_dict):,} | dim={vocab_dim}")
    return index_dict, word_vectors, combined

def get_data(index_dict, word_vectors, combined):
    n_symbols = len(index_dict) + 1
    embedding_weights = np.zeros((n_symbols, vocab_dim))
    for word, index in index_dict.items():
        embedding_weights[index, :] = word_vectors[word]
    log.info(f"[Embedding] n_symbols={n_symbols:,} | weights={embedding_weights.shape}")
    return n_symbols, embedding_weights


# -----------------------------
# Modelo LSTM (igual lógica)
# -----------------------------
def train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test, x_val, y_val):
    tf.set_random_seed(seed_value)

    model = Sequential()
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))
    model.add(Bidirectional(LSTM(output_dim=128, activation='sigmoid')))
    model.add(Dropout(0.2))
    model.add(Dense(5, activation='sigmoid'))

    log.info("Compilando modelo…")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    log.info(f"Train: x={np.shape(x_train)} y={np.shape(y_train)} | "
             f"Val: x={np.shape(x_val)} y={np.shape(y_val)} | "
             f"Test: x={np.shape(x_test)} y={np.shape(y_test)}")

    log.info("Entrenando…")
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=n_epoch,
                        verbose=1,
                        validation_data=(x_val, y_val))

    base_min = optimal_epoch(history)
    log.info("Evaluando en test…")
    score = model.evaluate(x_test, y_test, batch_size=batch_size)

    # Guardado (mismo comportamiento/paths)
    yaml_string = model.to_yaml()
    with open(model_location + '.yml', 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    model.save_weights(model_location + '.h5')
    np.set_printoptions(threshold=sys.maxsize)
    log.info(f"[Modelo] Guardado YAML en: {model_location + '.yml'}")
    log.info(f"[Modelo] Guardado PESOS en: {model_location + sys.argv[1] + '.h5'}")

    prob_predicted = model.predict(x_test, verbose=1)
    label_predicted = Helper.predict_prob_encoder(prob_predicted)
    num_y_test = Helper.pd_encoder(y_test)
    num_y_predicted = Helper.pd_encoder(label_predicted)

    val_accuracy = accuracy_score(y_test, label_predicted)
    log.info(f"Accuracy final (test): {val_accuracy:.4f}")
    Helper.class_accuracy(y_test, label_predicted)

    with open(model_location + '_target.txt', 'wt') as f:
        for y in y_test:
            f.write(str(y) + '\n')
    with open(model_location + '_predicted.txt', 'wt') as f:
        for y in label_predicted:
            f.write(str(y) + '\n')
    log.info(f"[Salida] Targets: {model_location + '_target.txt'}")
    log.info(f"[Salida] Predichos: {model_location + '_predicted.txt'}")

    return [val_accuracy]


def get_FP_FN(label_predicted, label_target):
    FP_id_list = []
    FN_id_list = []
    for i in range(0, len(label_predicted)):
        if int(label_predicted[i]) == 1 and int(label_target[i]) == 0:
            FP_id_list.append(i)
        elif int(label_predicted[i]) == 0 and int(label_target[i]) == 1:
            FN_id_list.append(i)

    with open('model_block' + '/labels/list/lstm_FP_' + sys.argv[1] + '.txt', 'wt') as f:
        for fp in FP_id_list:
            f.write(str(test_list[int(fp)]) + '\n')
    with open('model_block' + '/labels/list/lstm_FN_' + sys.argv[1] + '.txt', 'wt') as f:
        for fn in FN_id_list:
            f.write(str(test_list[int(fn)]) + '\n')


# -----------------------------
# Pipeline (igual lógica)
# -----------------------------
def train():
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    rn.seed(seed_value)

    log.info("=== PIPELINE: inicio ===")
    with PhaseTimer("Carga y split"):
        combined, y, x_train, x_val, x_test, y_train, y_val, y_test = loadfile()

    with PhaseTimer("Tokenización"):
        combined = tokenizer(combined)
        x_train = tokenizer(x_train)
        x_test = tokenizer(x_test)
        x_val = tokenizer(x_val)

    with PhaseTimer("Word2Vec + diccionarios"):
        index_dict, word_vectors, combined = word2vec_train(combined)
        x_train = input_transform(x_train)
        x_test = input_transform(x_test)
        x_val = input_transform(x_val)

    with PhaseTimer("Preparar embedding"):
        n_symbols, embedding_weights = get_data(index_dict, word_vectors, combined)

    with PhaseTimer("Entrenamiento LSTM"):
        result = train_lstm(n_symbols, embedding_weights, x_train, y_train, x_val, y_val, x_test, y_test)

    log.info("=== PIPELINE: fin ===")
    return result


def pipeline_train(iterations):
    seed_and_result = {}
    if iterations == 1:
        return train()
    else:
        for i in range(0, iterations):
            log.info(f"Iteración: {i}")
            global seed_value
            result = train()
            seed_and_result[seed_value] = result
            seed_value = seed_value + seed_window
            i = i + 1
        return seed_and_result


def eval_metric(model, history, metric_name):
    metric = history.history[metric_name]
    val_metric = history.history['val_' + metric_name]
    e = range(1, n_epoch + 1)
    plt.plot(e, metric, 'bo', label='Train ' + metric_name)
    plt.plot(e, val_metric, 'b', label='Validation ' + metric_name)
    plt.xlabel('Epoch number')
    plt.ylabel(metric_name)
    plt.title('Comparing training and validation ' + metric_name + ' for ' + model.name)
    plt.legend()
    plt.show()


def optimal_epoch(model_hist):
    min_epoch = np.argmin(model_hist.history['val_loss']) + 1
    log.info("Minimum validation loss reached in epoch {}".format(min_epoch))
    return min_epoch


# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    log.info(f"Script: {os.path.abspath(__file__)}")
    log.info(f"Dataset arg: {sys.argv[1] if len(sys.argv)>1 else '(faltante)'}")
    result_dict = pipeline_train(n_iterations)
    log.info(f"Dataset: {sys.argv[1]}")
