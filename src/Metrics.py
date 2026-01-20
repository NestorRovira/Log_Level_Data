from keras.callbacks import Callback
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, balanced_accuracy_score, accuracy_score
from sklearn.utils import resample

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_auc = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(
            self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        pos_label=1
        _val_f1 = f1_score(val_targ, val_predict, labels=[pos_label],pos_label=1, average ='binary')
        _val_recall = recall_score(val_targ, val_predict, labels=[pos_label],pos_label=1, average ='binary')
        _val_precision = precision_score(val_targ, val_predict, labels=[pos_label],pos_label=1, average ='binary')
        _val_auc = roc_auc_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        self.val_auc.append(_val_auc)
        return

import numpy as np

LEVELS = ['trace','debug','info','warn','error']

def ordinal_vec_to_level(vec, thr: float = 0.5) -> int:
    """
    Convierte un vector ordinal de 5 probs (sigmoid) al índice de nivel.
    Ej: [0.9,0.7,0.6,0.2,0.1] -> [1,1,1,0,0] -> 2 (info)
    """
    binv, cut = [], False
    for p in vec:
        if not cut and p >= thr:
            binv.append(1)
        else:
            cut = True
            binv.append(0)
    return sum(binv) - 1

def aod(y_true_idx, y_pred_idx) -> float:
    """
    Average Ordinal Distance, en [0..1] (más alto es mejor).
    y_*: iterables de índices 0..4
    """
    y_true_idx = np.asarray(y_true_idx, dtype=int)
    y_pred_idx = np.asarray(y_pred_idx, dtype=int)
    maxdist = np.maximum(y_true_idx, 4 - y_true_idx)  # 4 porque hay 5 niveles
    dist = np.abs(y_true_idx - y_pred_idx)
    maxdist = np.where(maxdist == 0, 1, maxdist)      # evita /0
    return float(np.mean(1 - dist / maxdist))

