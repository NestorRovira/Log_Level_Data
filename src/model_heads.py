from keras.layers import Dense  # KERAS standalone (2.3.1)

def add_head_ordinal(x):
    """
    Cabeza ORDINAL (sigmoid + BCE).
    """
    out = Dense(5, activation='sigmoid', name='out_ordinal')(x)
    return out, 'binary_crossentropy', 'ordinal'

def add_head_onehot(x):
    """
    Cabeza ONE-HOT (softmax + CCE).
    """
    out = Dense(5, activation='softmax', name='out_onehot')(x)
    return out, 'categorical_crossentropy', 'onehot'
