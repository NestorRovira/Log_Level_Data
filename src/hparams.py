from dataclasses import dataclass
from typing import List, Optional

# 30 semillas predefinidas (mismo orden para A y B)
SEEDS: List[int] = [
    1001, 1013, 1021, 1033, 1061, 1091, 1123, 1151, 1181, 1213,
    1237, 1277, 1291, 1301, 1327, 1361, 1381, 1423, 1439, 1451,
    1471, 1493, 1523, 1553, 1583, 1601, 1621, 1657, 1693, 1709,
]

@dataclass(frozen=True)
class HParams:
    # Modelo & training
    embedding_dim: int = 100
    rnn_units: int = 128
    dropout: float = 0.2
    epochs: int = 100
    batch_size: int = 24
    learning_rate: float = 1e-3  # en Keras 2.3.1 usa 'lr' al crear el optimizador
    # Datos
    max_len: Optional[int] = 100     # si None, se infiere del X.shape[1]
    # Splits rápidos (si no usas ficheros de split guardados)
    val_split: float = 0.2
    test_split: float = 0.2
    stratified: bool = True
    # Embeddings
    trainable_embeddings: bool = False  # True si quieres fine-tuning

HP = HParams()
