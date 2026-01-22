# -*- coding: utf-8 -*-
import os
import json
import argparse
import logging
import secrets
from datetime import datetime

import pandas as pd

from src.block_level_LSTM import train_and_eval_once, ensure_dir, flatten_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("DeepLV-Runner")

def now_utc():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def generate_random_seeds(n):
    seeds = set()
    while len(seeds) < n:
        seeds.add(secrets.randbelow(2_000_000_000) + 1)
    return list(seeds)

def summarize_so_far(df):
    cols = ["test_set", "model_type", "accuracy", "auc", "aod"]
    if df is None or len(df) == 0:
        return None
    d = df[cols].dropna(subset=["accuracy"]).copy()
    if len(d) == 0:
        return None
    return d.groupby(["test_set", "model_type"], as_index=False).agg(
        n=("accuracy", "count"),
        acc_mean=("accuracy", "mean"),
        auc_mean=("auc", "mean"),
        aod_mean=("aod", "mean"),
        acc_std=("accuracy", "std"),
        auc_std=("auc", "std"),
        aod_std=("aod", "std"),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, default="cassandra")
    ap.add_argument("--tests", type=str, default="karaf,wicket")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--out_root", type=str, default="results")
    ap.add_argument("--exp_name", type=str, default=None)
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

    test_ds = [x.strip() for x in args.tests.split(",") if x.strip() != ""]
    exp_name = args.exp_name if args.exp_name else f"experiment_{args.train}"
    exp_dir = ensure_dir(os.path.join(args.out_root, exp_name))
    models_dir = ensure_dir(os.path.join(exp_dir, "models"))

    plan_path = os.path.join(exp_dir, "execution_plan.json")
    jsonl_path = os.path.join(exp_dir, "results.jsonl")
    csv_path = os.path.join(exp_dir, "results.csv")

    if os.path.exists(plan_path):
        with open(plan_path, "r", encoding="utf-8") as f:
            plan_obj = json.load(f)
        plan = plan_obj.get("plan", [])
        log.info("Plan existente cargado: %s", plan_path)
    else:
        seeds = generate_random_seeds(args.runs)
        plan = []
        for s in seeds:
            order = ["onehot", "ordinal"]
            if secrets.randbelow(2) == 0:
                order = ["ordinal", "onehot"]
            plan.append({"pair_seed": int(s), "order": order})
        plan_obj = {
            "created_utc": now_utc(),
            "train": args.train,
            "tests": test_ds,
            "runs": args.runs,
            "plan": plan,
            "hparams": {
                "max_len": args.max_len,
                "embed_dim": args.embed_dim,
                "w2v_window": args.w2v_window,
                "w2v_min_count": args.w2v_min_count,
                "w2v_workers": args.w2v_workers,
                "w2v_sg": args.w2v_sg,
                "lstm_units": args.lstm_units,
                "dropout": args.dropout,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs
            }
        }
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan_obj, f, ensure_ascii=False, indent=2)
        log.info("Plan guardado: %s", plan_path)

    log.info("=== EXPERIMENTO ===")
    log.info("train=%s tests=%s runs=%s out=%s", args.train, test_ds, len(plan), exp_dir)

    df_all = None
    if os.path.exists(csv_path):
        try:
            df_all = pd.read_csv(csv_path)
        except Exception:
            df_all = None

    for i, item in enumerate(plan, start=1):
        seed = int(item["pair_seed"])
        order = item["order"]
        log.info("Par %d/%d | seed=%s | order=%s", i, len(plan), seed, order)

        for j, model_type in enumerate(order):
            log.info("Ejecución | model=%s | seed=%s", model_type, seed)

            res = train_and_eval_once(
                model_type=model_type,
                train_dataset=args.train,
                test_datasets=test_ds,
                seed=seed,
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

            res["execution"] = {"pair_seed": seed, "order_in_pair": int(j), "pair_order": order}

            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")

            rows = flatten_results(res)
            df_rows = pd.DataFrame(rows)
            df_rows["pair_seed"] = seed
            df_rows["order_in_pair"] = int(j)
            df_rows["pair_order"] = " > ".join(order)

            if os.path.exists(csv_path):
                df_rows.to_csv(csv_path, mode="a", header=False, index=False)
            else:
                df_rows.to_csv(csv_path, index=False)

            for ts, m in res["metrics"].items():
                log.info(
                    "MÉTRICAS | model=%s | %s | acc=%.4f auc=%.4f aod=%.4f n=%s",
                    model_type,
                    ts,
                    m.get("accuracy", float("nan")),
                    m.get("auc", float("nan")),
                    m.get("aod", float("nan")),
                    m.get("n", None),
                )

            try:
                df_all = pd.read_csv(csv_path)
                summ = summarize_so_far(df_all)
                if summ is not None:
                    log.info("RESUMEN (media) hasta ahora:")
                    for _, r in summ.iterrows():
                        log.info(
                            "  %s | %s | n=%d acc=%.4f auc=%.4f aod=%.4f",
                            r["test_set"],
                            r["model_type"],
                            int(r["n"]),
                            float(r["acc_mean"]),
                            float(r["auc_mean"]),
                            float(r["aod_mean"])
                        )
            except Exception:
                pass

    log.info("=== FIN EXPERIMENTO ===")
    log.info("Resultados CSV: %s", csv_path)
    log.info("Resultados JSONL: %s", jsonl_path)
    log.info("Plan: %s", plan_path)
    log.info("Modelos: %s", models_dir)

if __name__ == "__main__":
    main()
