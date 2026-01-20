# -*- coding: utf-8 -*-
# ================================================================
# block_level_LSTM.py  (versión TF1.15 + Keras 2.3.1 + gensim 3.8.3)
# ------------------------------------------------
# - Soporta dos cabezas: --model {ordinal|onehot}
# - Busca logged_syn_<dataset>.csv en rutas conocidas
# - Logs detallados por fase + medición de tiempo
# ================================================================

import os
import sys
import re
import ast
import time
import logging
import random as rn
import argparse
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

# ===== TensorFlow 1.x backend para Keras =====
import tensorflow as tf
try:
    # Asegura modo gráfico clásico (por si el entorno hubiera tocado eager)
    tf.compat.v1.disable_eager_execution()
except Exception:
    pass

from keras import backend as K
from keras.layers import Input, Embedding, Bidirectional, LSTM, Dropout
from keras.models import Model
from keras.optimizers import Adam

# gensim 3.8.3
from gensim.models import Word2Vec

# sklearn
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("block_LSTM")

# ---------------------------------------------------------------------
# PhaseTimer: intenta traer de Helper, si no existe usa fallback
# ---------------------------------------------------------------------
try:
    from Helper import PhaseTimer as _PhaseTimer  # type: ignore
except Exception:
    class _PhaseTimer:
        def __init__(self, label: str):
            self.label = label
        def __enter__(self):
            self.t0 = time.perf_counter()
            log.info(f"[{self.label}] inicio")
            return self
        def __exit__(self, exc_type, exc, tb):
            dt = time.perf_counter() - self.t0
            if exc:
                log.error(f"[{self.label}] ERROR tras {dt:.2f}s: {exc}")
            else:
                log.info(f"[{self.label}] fin en {dt:.2f}s")
            return False
PhaseTimer = _PhaseTimer

# ---------------------------------------------------------------------
# Hiperparámetros y cabezas
# ---------------------------------------------------------------------
from hparams import HP
from model_heads import add_head_ordinal, add_head_onehot

# ---------------------------------------------------------------------
# Semillas / determinismo (TF1 + Keras 2.3.1)
# ---------------------------------------------------------------------
def set_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    rn.seed(seed)
    np.random.seed(seed)
    try:
        tf.set_random_seed(seed)  # TF1.x
    except Exception:
        try:
            tf.random.set_seed(seed)  # fallback por si el wrapper expone TF2 API
        except Exception:
            log.warning("No pude fijar semilla de TensorFlow.")

    import multiprocessing
    n = multiprocessing.cpu_count()
    cfg = tf.ConfigProto(intra_op_parallelism_threads=n,
                         inter_op_parallelism_threads=n)
    sess = tf.Session(config=cfg)
    K.set_session(sess)

# ---------------------------------------------------------------------
# Utilidades de etiquetas (ordinal vs onehot)
# ---------------------------------------------------------------------
LEVEL_ORDER = ["trace", "debug", "info", "warn", "error"]

def norm_level(x: str) -> str:
    x = (x or "").strip().lower()
    if x == "warning":
        return "warn"
    if x not in LEVEL_ORDER:
        return "info"
    return x

def level_to_index(x: str) -> int:
    return LEVEL_ORDER.index(norm_level(x))

def labels_to_ordinal(idx: np.ndarray) -> np.ndarray:
    idx = np.asarray(idx).astype(int)
    y = np.zeros((len(idx), 5), dtype="float32")
    for i, k in enumerate(idx):
        y[i, :k+1] = 1.0
    return y

def labels_to_onehot(idx: np.ndarray) -> np.ndarray:
    idx = np.asarray(idx).astype(int)
    y = np.zeros((len(idx), 5), dtype="float32")
    y[np.arange(len(idx)), idx] = 1.0
    return y

def ordinal_to_index(y_ord: np.ndarray) -> np.ndarray:
    return np.sum((y_ord >= 0.5).astype(int), axis=1) - 1

def ensure_label_format(y, model_kind: str) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 1:
        return labels_to_ordinal(y) if model_kind == "ordinal" else labels_to_onehot(y)
    if y.ndim == 2 and y.shape[1] == 5:
        if model_kind == "ordinal":
            return y
        else:
            idx = ordinal_to_index(y)
            return labels_to_onehot(idx)
    raise ValueError("Formato de etiquetas no reconocido.")

# ---------------------------------------------------------------------
# Carga de datos desde el CSV de block_processing
# ---------------------------------------------------------------------
CANDIDATE_PATHS = [
    os.path.join("block_processing", "blocks", "logged_syn_{ds}.csv"),
    os.path.join("blocks", "logged_syn_{ds}.csv"),
]

CANDIDATE_TEXT_COLS = ["Values", "values", "Tokens", "tokens", "Content", "content", "Text", "text"]
CANDIDATE_LABEL_COLS = ["Level", "level", "LogLevel", "log_level", "Label", "label", "lvl"]

def _find_csv(dataset: Optional[str]) -> str:
    ds = dataset or "cassandra"
    for tmpl in CANDIDATE_PATHS:
        p = tmpl.format(ds=ds)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"No encuentro el CSV de bloques con log para dataset='{ds}'. "
        f"Probadas rutas: {[t.format(ds=ds) for t in CANDIDATE_PATHS]}"
    )

def _pick_column(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"No se encontró ninguna de las columnas esperadas. Vistas={list(df.columns)}; "
        f"Buscadas={candidates}"
    )

def _parse_tokens(cell) -> List[str]:
    if isinstance(cell, list):
        return [str(t) for t in cell]
    if isinstance(cell, str):
        s = cell.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                li = ast.literal_eval(s)
                return [str(t) for t in li]
            except Exception:
                pass
        return [tok for tok in re.split(r"[^A-Za-z0-9_]+", s) if tok]
    return [str(cell)]

def loadfile(dataset: Optional[str]):
    with PhaseTimer("Carga CSV"):
        path = _find_csv(dataset)
        df = pd.read_csv(path, encoding="utf-8")
        log.info(f"CSV: {path} | filas={len(df):,} | columnas={list(df.columns)}")

    text_col = _pick_column(df, CANDIDATE_TEXT_COLS)
    label_col = _pick_column(df, CANDIDATE_LABEL_COLS)

    with PhaseTimer("Parseo tokens + labels"):
        tokens_all = [_parse_tokens(v) for v in df[text_col].tolist()]
        labels_idx = np.array([level_to_index(str(v)) for v in df[label_col].tolist()], dtype=np.int32)

    with PhaseTimer("Split 60/20/20 estratificado"):
        X_train, X_tmp, y_train, y_tmp = train_test_split(
            tokens_all, labels_idx, test_size=HP.val_split + HP.test_split,
            stratify=labels_idx, random_state=17020
        )
        rel_test = HP.test_split / (HP.val_split + HP.test_split)
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp, test_size=rel_test,
            stratify=y_tmp, random_state=17021
        )
        log.info(f"Split -> train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    combined = X_train + X_val + X_test
    y_all = labels_idx
    return combined, y_all, X_train, X_val, X_test, y_train, y_val, y_test

# ---------------------------------------------------------------------
# Tokenizer “identidad” (ya venimos tokenizados)
# ---------------------------------------------------------------------
def tokenizer(token_lists: List[List[str]]) -> List[List[str]]:
    return token_lists

# ---------------------------------------------------------------------
# Word2Vec + diccionario + transformaciones (gensim 3.8.3 compatible)
# ---------------------------------------------------------------------
def word2vec_train(tokenized_sentences: List[List[str]],
                   vector_size: int = None, window: int = 5, min_count: int = 1):
    """
    Compatibilidad gensim 3.x y 4.x:
      - 4.x: Word2Vec(..., vector_size=dim)
      - 3.x: Word2Vec(..., size=dim)
      - vocab: 4.x -> wv.index_to_key ; 3.x -> wv.index2word
    """
    dim = vector_size or HP.embedding_dim
    try:
        # gensim 4.x
        model = Word2Vec(sentences=tokenized_sentences, vector_size=dim,
                         window=window, min_count=min_count, workers=4)
        vocab_list = model.wv.index_to_key
    except TypeError:
        # gensim 3.x
        model = Word2Vec(sentences=tokenized_sentences, size=dim,
                         window=window, min_count=min_count, workers=4)
        vocab_list = model.wv.index2word

    index_dict = {w: i + 1 for i, w in enumerate(vocab_list)}
    log.info(f"Word2Vec entrenado. vocab={len(index_dict)} dim={dim}")
    return index_dict, model.wv, tokenized_sentences

def input_transform(tokenized_sentences: List[List[str]], index_dict: dict,
                    pad_to: Optional[int] = None) -> np.ndarray:
    seqs = [[index_dict.get(tok, 0) for tok in sent] for sent in tokenized_sentences]
    maxlen = pad_to or (max((len(s) for s in seqs), default=1))
    X = np.zeros((len(seqs), maxlen), dtype=np.int32)
    for i, s in enumerate(seqs):
        L = min(len(s), maxlen)
        X[i, :L] = s[:L]
    return X

def get_data(index_dict: dict, word_vectors, _combined) -> Tuple[int, np.ndarray]:
    """
    Construye matriz de embeddings compatible con gensim 3.x y 4.x.
    """
    n_symbols = len(index_dict) + 1
    if hasattr(word_vectors, "vector_size"):
        dim = word_vectors.vector_size
    else:
        # gensim 3.x
        dim = word_vectors.syn0.shape[1]
    emb = np.zeros((n_symbols, dim), dtype=np.float32)
    for w, i in index_dict.items():
        if w in word_vectors:
            emb[i, :] = word_vectors[w]
    log.info(f"Matriz embeddings: shape={emb.shape}")
    return n_symbols, emb

# ---------------------------------------------------------------------
# Backbone (Embedding -> BiLSTM -> Dropout)
# ---------------------------------------------------------------------
def build_backbone(vocab_size: int, embedding_weights: Optional[np.ndarray], max_len: int):
    emb_dim = HP.embedding_dim if embedding_weights is None else embedding_weights.shape[1]
    inp = Input(shape=(max_len,), name="tokens")
    if embedding_weights is None:
        emb = Embedding(vocab_size, emb_dim,
                        name="emb",
                        trainable=True,
                        input_length=max_len,
                        mask_zero=True)(inp)
    else:
        emb = Embedding(input_dim=vocab_size,
                        output_dim=emb_dim,
                        weights=[embedding_weights],
                        trainable=HP.trainable_embeddings,
                        name="emb",
                        input_length=max_len,
                        mask_zero=True)(inp)
    x = Bidirectional(LSTM(HP.rnn_units,
                           activation='sigmoid',
                           return_sequences=False),
                      name="bilstm")(emb)
    x = Dropout(HP.dropout, name="dropout")(x)
    return inp, x

# ---------------------------------------------------------------------
# Entrenamiento de UNA corrida
# ---------------------------------------------------------------------
def train_one(model_kind: str, dataset: Optional[str] = None, seed: int = 1001):
    set_seeds(seed)

    # 1) Carga + split
    combined, y_all, X_tr, X_va, X_te, y_tr_idx, y_va_idx, y_te_idx = loadfile(dataset)
    log.info(f"Datos: train={len(X_tr)} val={len(X_va)} test={len(X_te)}")

    # 2) Tokenización (identidad) + Word2Vec + diccionarios
    with PhaseTimer("Tokenización"):
        combined_tok = tokenizer(combined)
        Xtr_tok = tokenizer(X_tr)
        Xva_tok = tokenizer(X_va)
        Xte_tok = tokenizer(X_te)

    with PhaseTimer("Word2Vec + diccionario"):
        index_dict, wv, combined_tok = word2vec_train(combined_tok)
        pad_len = HP.max_len or max((len(s) for s in Xtr_tok), default=1)
        Xtr_idx = input_transform(Xtr_tok, index_dict, pad_to=pad_len)
        Xva_idx = input_transform(Xva_tok, index_dict, pad_to=pad_len)
        Xte_idx = input_transform(Xte_tok, index_dict, pad_to=pad_len)

    with PhaseTimer("Embedding matrix"):
        n_symbols, embedding_weights = get_data(index_dict, wv, combined_tok)

    # 3) Etiquetas al formato correcto
    y_tr = ensure_label_format(y_tr_idx, model_kind)
    y_va = ensure_label_format(y_va_idx, model_kind)
    y_te = ensure_label_format(y_te_idx, model_kind)

    # 4) Modelo (KERAS standalone)
    with PhaseTimer("Construir modelo"):
        inp, x = build_backbone(n_symbols, embedding_weights, pad_len)
        if model_kind == "ordinal":
            out, loss_name, kind = add_head_ordinal(x)
        else:
            out, loss_name, kind = add_head_onehot(x)
        model = Model(inp, out, name=f"BiLSTM_{kind}")
        opt = Adam(lr=HP.learning_rate)  # Keras 2.3.1 usa 'lr'
        model.compile(optimizer=opt, loss=loss_name, metrics=["accuracy"])
        model.summary(print_fn=lambda s: log.info(s))

    # 5) Entrenamiento
    with PhaseTimer(f"Entrenamiento ({kind})"):
        t0 = time.perf_counter()
        hist = model.fit(
            Xtr_idx, y_tr,
            validation_data=(Xva_idx, y_va),
            epochs=HP.epochs,
            batch_size=HP.batch_size,
            verbose=2,
        )
        train_time = time.perf_counter() - t0
    log.info(f"Tiempo entrenamiento: {train_time:.1f}s | última val_acc={hist.history['val_accuracy'][-1]:.4f}")

    # 6) Evaluación
    with PhaseTimer("Evaluación test"):
        test_loss, test_acc = model.evaluate(Xte_idx, y_te, verbose=0)
    log.info(f"TEST -> loss={test_loss:.4f} | acc={test_acc:.4f}")

    return {"loss": float(test_loss), "accuracy": float(test_acc), "time_s": float(train_time)}

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="cassandra",
                        help="Nombre del dataset (p.ej., cassandra). Busca logged_syn_<dataset>.csv")
    parser.add_argument("--model", choices=["ordinal", "onehot"], default="ordinal",
                        help="Cabeza de salida: 'ordinal' (sigmoid+BCE) o 'onehot' (softmax+CCE)")
    parser.add_argument("--seed", type=int, default=1001)
    args = parser.parse_args()

    log.info(f"=== INICIO: model={args.model} | seed={args.seed} | dataset={args.dataset} ===")
    try:
        result = train_one(args.model, dataset=args.dataset, seed=args.seed)
        log.info(f"=== FIN {args.model.upper()} | Resultados: {result} ===")
    except Exception as e:
        log.exception(f"Fallo de ejecución: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
