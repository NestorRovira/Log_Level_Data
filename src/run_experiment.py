import subprocess, sys, time, pathlib, logging, os
from hparams import SEEDS

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("orchestrator")

PY = sys.executable
RUNNER = str(pathlib.Path(__file__).resolve().parent / "block_level_LSTM.py")

def run_one(model_kind: str, seed: int, dataset: str):
    cmd = [PY, RUNNER]
    if dataset:
        cmd.append(dataset)  # arg posicional si lo usas (p.ej. 'cassandra')
    cmd += ["--model", model_kind, "--seed", str(seed)]
    t0 = time.perf_counter()
    log.info(f"Lanzando: {cmd}")
    ret = subprocess.call(cmd)
    dt = time.perf_counter() - t0
    log.info(f"Fin: model={model_kind} seed={seed} exit={ret} tiempo={dt:.1f}s")
    return ret

if __name__ == "__main__":
    DATASET = os.environ.get("LOGLEVEL_DATASET", "cassandra")
    log.info(f"=== Experimento alternado A(ordinal) ↔ B(onehot) | dataset={DATASET} ===")
    for seed in SEEDS:
        run_one("ordinal", seed, DATASET)
        run_one("onehot",  seed, DATASET)
    log.info("=== Experimento completo ===")
