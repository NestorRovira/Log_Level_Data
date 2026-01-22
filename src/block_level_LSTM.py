# -*- coding: utf-8 -*-
import os
import sys
import json
import time
import uuid
import argparse
import logging
import platform
import csv
from datetime import datetime

import numpy as np
import pandas as pd
import random as rn

import tensorflow as tf
from gensim.models.word2vec import Word2Vec

from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences

from src import Metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("DeepLV")

def set_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    rn.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

def now_utc():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def find_csv_for_dataset(dataset):
    candidates = [
        os.path.join("block_processing", "blocks", f"logged_syn_{dataset}.csv"),
        os.path.join("blocks", f"logged_syn_{dataset}.csv"),
        os.path.join("block_processing", "blocks", f"logged_syn_{dataset.lower()}.csv"),
        os.path.join("blocks", f"logged_syn_{dataset.lower()}.csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("No se encontró el CSV del dataset: %s (probé %s)" % (dataset, candidates))

def normalize_text(s):
    if s is None:
        return ""
    t = str(s)
    t = t.replace("'", "").replace('"', "")
    t = t.replace("[", " ").replace("]", " ")
    t = t.replace(",", " ")
    t = t.replace("\\n", " ").replace("\\t", " ")
    t = t.replace("\n", " ").replace("\t", " ")
    return t

def word_splitter(word, out):
    import re
    w = str(word)
    splitted = re.sub("([A-Z][a-z]+)", r" \1", re.sub("([A-Z]+)", r" \1", w)).split()
    for x in splitted:
        out.append(x.lower())

def tokenize_docs(docs):
    out = []
    for doc in docs:
        docText = []
        t = normalize_text(doc)
        parts = [p for p in t.split(" ") if p.strip() != ""]
        for w in parts:
            word_splitter(w, docText)
        out.append(docText)
    return out

def build_w2v(sentences, embed_dim, window, min_count, workers, sg, seed):
    model = Word2Vec(
        sentences=sentences,
        size=embed_dim,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        seed=seed
    )
    return model

def build_vocab_and_embeddings(w2v_model):
    vocab = list(w2v_model.wv.vocab.keys())
    w2idx = {}
    idx2w = {}
    for i, w in enumerate(vocab, start=1):
        w2idx[w] = i
        idx2w[i] = w
    n_symbols = len(w2idx) + 1
    embed_dim = w2v_model.vector_size
    W = np.zeros((n_symbols, embed_dim), dtype=np.float32)
    for w, i in w2idx.items():
        W[i] = w2v_model.wv[w]
    return w2idx, idx2w, W

def docs_to_sequences(tokenized_docs, w2idx):
    seqs = []
    oov = 0
    total = 0
    for sent in tokenized_docs:
        s = []
        for w in sent:
            total += 1
            if w in w2idx:
                s.append(w2idx[w])
            else:
                s.append(0)
                oov += 1
        seqs.append(s)
    oov_rate = float(oov) / float(total) if total > 0 else 0.0
    return seqs, oov_rate

def stratified_split(X_docs, y_int, seed):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_docs, y_int, test_size=0.20, random_state=seed, stratify=y_int
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=seed, stratify=y_train
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def build_model(model_type, n_symbols, embedding_weights, max_len, lstm_units, dropout, lr):
    model = Sequential()
    model.add(
        Embedding(
            input_dim=n_symbols,
            output_dim=embedding_weights.shape[1],
            weights=[embedding_weights],
            input_length=max_len,
            mask_zero=True,
            trainable=True
        )
    )
    model.add(Bidirectional(LSTM(units=lstm_units, activation="sigmoid", return_sequences=False)))
    model.add(Dropout(dropout))
    if model_type == "onehot":
        model.add(Dense(5, activation="softmax"))
        model.compile(optimizer=Adam(lr=lr), loss="categorical_crossentropy", metrics=["accuracy"])
    else:
        model.add(Dense(5, activation="sigmoid"))
        model.compile(optimizer=Adam(lr=lr), loss="binary_crossentropy", metrics=["binary_accuracy"])
    return model

def get_versions():
    import pkg_resources
    pkgs = ["tensorflow", "Keras", "gensim", "numpy", "pandas", "scikit-learn", "scipy"]
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for p in pkgs:
        try:
            out[p] = pkg_resources.get_distribution(p).version
        except Exception:
            out[p] = None
    return out

def train_and_eval_once(
    model_type,
    train_dataset,
    test_datasets,
    seed,
    max_len,
    embed_dim,
    w2v_window,
    w2v_min_count,
    w2v_workers,
    w2v_sg,
    lstm_units,
    dropout,
    lr,
    batch_size,
    epochs,
    models_dir
):
    set_seeds(seed)
    run_id = str(uuid.uuid4())
    t0 = time.time()

    csv.field_size_limit(2147483647)

    train_csv = find_csv_for_dataset(train_dataset)
    df_train = pd.read_csv(train_csv, usecols=["Key", "Values", "Level", "Message"], engine="python")
    y_train_int_full = Metrics.levels_to_int(df_train["Level"].values.tolist())
    X_docs_full = df_train["Values"].values.tolist()

    X_train_docs, X_val_docs, X_test_docs, y_train_int, y_val_int, y_test_int = stratified_split(X_docs_full, y_train_int_full, seed)

    tok_train = tokenize_docs(X_train_docs)
    tok_val = tokenize_docs(X_val_docs)
    tok_test = tokenize_docs(X_test_docs)

    tok_for_w2v = tok_train + tok_val + tok_test
    w2v = build_w2v(tok_for_w2v, embed_dim=embed_dim, window=w2v_window, min_count=w2v_min_count, workers=w2v_workers, sg=w2v_sg, seed=seed)
    w2idx, idx2w, W = build_vocab_and_embeddings(w2v)

    seq_train, oov_train = docs_to_sequences(tok_train, w2idx)
    seq_val, oov_val = docs_to_sequences(tok_val, w2idx)
    seq_test, oov_test = docs_to_sequences(tok_test, w2idx)

    X_train_mat = pad_sequences(seq_train, maxlen=max_len)
    X_val_mat = pad_sequences(seq_val, maxlen=max_len)
    X_test_mat = pad_sequences(seq_test, maxlen=max_len)

    if model_type == "onehot":
        y_train = Metrics.to_onehot(y_train_int, n_classes=5)
        y_val = Metrics.to_onehot(y_val_int, n_classes=5)
        y_test = Metrics.to_onehot(y_test_int, n_classes=5)
    else:
        y_train = Metrics.ordinal_targets_from_int(y_train_int, n_classes=5)
        y_val = Metrics.ordinal_targets_from_int(y_val_int, n_classes=5)
        y_test = Metrics.ordinal_targets_from_int(y_test_int, n_classes=5)

    model = build_model(
        model_type=model_type,
        n_symbols=W.shape[0],
        embedding_weights=W,
        max_len=max_len,
        lstm_units=lstm_units,
        dropout=dropout,
        lr=lr
    )

    t_train0 = time.time()
    model.fit(
        X_train_mat,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val_mat, y_val),
        verbose=2
    )
    t_train1 = time.time()

    ensure_dir(models_dir)
    model_path = os.path.join(models_dir, f"model_{train_dataset}_{model_type}_seed{seed}.h5")
    model.save(model_path)

    results = {
        "run_id": run_id,
        "timestamp_utc": now_utc(),
        "train_dataset": train_dataset,
        "test_datasets": list(test_datasets),
        "model_type": model_type,
        "seed": int(seed),
        "paths": {"train_csv": train_csv, "saved_model": model_path},
        "sizes": {
            "train": int(len(X_train_docs)),
            "val": int(len(X_val_docs)),
            "test_internal": int(len(X_test_docs)),
            "vocab_size": int(W.shape[0]),
            "embed_dim": int(W.shape[1]),
        },
        "oov_rates": {"train": float(oov_train), "val": float(oov_val), "test_internal": float(oov_test)},
        "hparams": {
            "max_len": int(max_len),
            "embed_dim": int(embed_dim),
            "w2v_window": int(w2v_window),
            "w2v_min_count": int(w2v_min_count),
            "w2v_workers": int(w2v_workers),
            "w2v_sg": int(w2v_sg),
            "lstm_units": int(lstm_units),
            "dropout": float(dropout),
            "lr": float(lr),
            "batch_size": int(batch_size),
            "epochs": int(epochs),
        },
        "versions": get_versions(),
        "timing_s": {"train": float(t_train1 - t_train0), "total": float(time.time() - t0)},
        "metrics": {}
    }

    t_eval0 = time.time()
    if model_type == "onehot":
        proba_int = model.predict(X_test_mat, batch_size=batch_size, verbose=0)
        pred_int = np.argmax(proba_int, axis=1).astype(np.int64)
    else:
        ord_prob = model.predict(X_test_mat, batch_size=batch_size, verbose=0)
        pred_int = Metrics.decode_ordinal_to_int(ord_prob, threshold=0.5)
        proba_int = Metrics.ordinal_cumprob_to_classprob(ord_prob)

    m_internal = Metrics.compute_all_metrics(y_test_int, pred_int, proba_int)
    results["metrics"]["test_internal"] = m_internal
    results["metrics"]["test_internal"]["n"] = int(len(X_test_docs))
    results["metrics"]["test_internal"]["oov_rate"] = float(oov_test)

    for ds in test_datasets:
        csv.field_size_limit(2147483647)
        test_csv = find_csv_for_dataset(ds)
        df_ext = pd.read_csv(test_csv, usecols=["Key", "Values", "Level", "Message"], engine="python")
        y_ext_int = Metrics.levels_to_int(df_ext["Level"].values.tolist())
        X_ext_docs = df_ext["Values"].values.tolist()
        tok_ext = tokenize_docs(X_ext_docs)
        seq_ext, oov_ext = docs_to_sequences(tok_ext, w2idx)
        X_ext_mat = pad_sequences(seq_ext, maxlen=max_len)

        if model_type == "onehot":
            proba_ext = model.predict(X_ext_mat, batch_size=batch_size, verbose=0)
            pred_ext_int = np.argmax(proba_ext, axis=1).astype(np.int64)
        else:
            ord_ext = model.predict(X_ext_mat, batch_size=batch_size, verbose=0)
            pred_ext_int = Metrics.decode_ordinal_to_int(ord_ext, threshold=0.5)
            proba_ext = Metrics.ordinal_cumprob_to_classprob(ord_ext)

        m_ext = Metrics.compute_all_metrics(y_ext_int, pred_ext_int, proba_ext)
        results["metrics"][ds] = dict(m_ext)
        results["metrics"][ds]["oov_rate"] = float(oov_ext)
        results["metrics"][ds]["n"] = int(len(X_ext_docs))
        results["paths"][f"test_csv_{ds}"] = test_csv

    t_eval1 = time.time()
    results["timing_s"]["eval"] = float(t_eval1 - t_eval0)
    results["timing_s"]["total"] = float(time.time() - t0)

    return results

def flatten_results(r):
    base = {
        "run_id": r["run_id"],
        "timestamp_utc": r["timestamp_utc"],
        "train_dataset": r["train_dataset"],
        "model_type": r["model_type"],
        "seed": r["seed"],
        "train_n": r["sizes"]["train"],
        "val_n": r["sizes"]["val"],
        "vocab_size": r["sizes"]["vocab_size"],
        "embed_dim": r["sizes"]["embed_dim"],
        "max_len": r["hparams"]["max_len"],
        "lstm_units": r["hparams"]["lstm_units"],
        "dropout": r["hparams"]["dropout"],
        "lr": r["hparams"]["lr"],
        "batch_size": r["hparams"]["batch_size"],
        "epochs": r["hparams"]["epochs"],
        "time_train_s": r["timing_s"]["train"],
        "time_eval_s": r["timing_s"]["eval"],
        "time_total_s": r["timing_s"]["total"],
        "tf": r["versions"].get("tensorflow"),
        "keras": r["versions"].get("Keras"),
        "gensim": r["versions"].get("gensim"),
        "numpy": r["versions"].get("numpy"),
        "pandas": r["versions"].get("pandas"),
        "sklearn": r["versions"].get("scikit-learn"),
    }
    rows = []
    for test_set, m in r["metrics"].items():
        row = dict(base)
        row["test_set"] = test_set
        row["accuracy"] = m.get("accuracy")
        row["auc"] = m.get("auc")
        row["aod"] = m.get("aod")
        row["test_n"] = m.get("n", None)
        row["oov_rate"] = m.get("oov_rate", None)
        rows.append(row)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_dataset", type=str)
    ap.add_argument("--model", type=str, default="ordinal", choices=["ordinal", "onehot"])
    ap.add_argument("--seed", type=int, default=17020)
    ap.add_argument("--test_datasets", type=str, default="karaf,wicket")
    ap.add_argument("--models_dir", type=str, default=os.path.join("results", "models"))
    ap.add_argument("--max_len", type=int, default=100)
    ap.add_argument("--embed_dim", type=int, default=100)
    ap.add_argument("--w2v_window", type=int, default=5)
    ap.add_argument("--w2v_min_count", type=int, default=1)
    ap.add_argument("--w2v_workers", type=int, default=4)
    ap.add_argument("--w2v_sg", type=int, default=1)
    ap.add_argument("--lstm_units", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()

    test_ds = [x.strip() for x in args.test_datasets.split(",") if x.strip() != ""]
    models_dir = ensure_dir(args.models_dir)

    log.info("=== INICIO ===")
    log.info("train_dataset=%s model=%s seed=%s test_datasets=%s models_dir=%s", args.train_dataset, args.model, args.seed, test_ds, models_dir)

    res = train_and_eval_once(
        model_type=args.model,
        train_dataset=args.train_dataset.strip(),
        test_datasets=test_ds,
        seed=args.seed,
        max_len=args.max_len,
        embed_dim=args.embed_dim,
        w2v_window=args.w2v_window,
        w2v_min_count=args.w2v_min_count,
        w2v_workers=args.w2v_workers,
        w2v_sg=args.w2v_sg,
        lstm_units=args.lstm_units,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        models_dir=models_dir
    )

    log.info("=== MÉTRICAS ===")
    for k, v in res["metrics"].items():
        log.info("%s | acc=%.4f auc=%.4f aod=%.4f n=%s", k, v.get("accuracy", float("nan")), v.get("auc", float("nan")), v.get("aod", float("nan")), v.get("n", None))

    log.info("=== FIN ===")

if __name__ == "__main__":
    main()
