import numpy as np
import pandas as pd
import tensorflow as tf
import chess
import os

def boardstate(fen):
    board = chess.Board(fen[0])
    fstr = str(fen[0])

    if board.has_kingside_castling_rights(chess.WHITE):
        WCKI = 1
    else:
        WCKI = 0
    if board.has_queenside_castling_rights(chess.WHITE):
        WCQ = 1
    else:
        WCQ = 0
    if board.is_check():
        WCH = 1
    else:
        WCH = 0

    if board.has_kingside_castling_rights(chess.BLACK):
        BCKI = 1
    else:
        BCKI = 0
    if board.has_queenside_castling_rights(chess.BLACK):
        BCQ = 1
    else:
        BCQ = 0
    if board.was_into_check():
        BCH = 1
    else:
        BCH = 0

    fw = [WCKI, WCQ, WCH]
    fb = [BCKI, BCQ, BCH]

    bstr = str(board)
    bstr = bstr.replace("p", "\ -1")
    bstr = bstr.replace("n", "\ -3")
    bstr = bstr.replace("b", "\ -4")
    bstr = bstr.replace("r", "\ -5")
    bstr = bstr.replace("q", "\ -9")
    bstr = bstr.replace("k", "\ -100")
    bstr = bstr.replace("P", "\ 1")
    bstr = bstr.replace("N", "\ 3")
    bstr = bstr.replace("B", "\ 4")
    bstr = bstr.replace("R", "\ 5")
    bstr = bstr.replace("Q", "\ 9")
    bstr = bstr.replace("K", "\ 100")
    bstr = bstr.replace(".", "\ 0")
    bstr = bstr.replace("\ ", ",")
    bstr = bstr.replace("'", " ")
    bstr = bstr.replace("\n", "")
    bstr = bstr.replace(" ", "")
    bstr = bstr[1:]
    bstr = eval(bstr)
    bstr = list(bstr)
    if "w" not in fstr:
        for i in range(len(bstr)):
            bstr[i] = bstr[i] * -1
        bstr.reverse()
        fs = fb
        fb = fw
        fw = fs

    BITBOARD = fw + fb + bstr

    return BITBOARD

def strfix(fen, tr):
    fstr = str(fen)

    if '#' in str(tr):
        if '-' in tr:
            t = -10000
        else:
            t = 10000
    elif '\ufeff+23' in str(tr):
        t = 0
    else:
        t = int(tr)

    if "w" not in fstr:
        t = t * -1

    t = t / 10

    return t

data = pd.read_csv('chessData.csv')

label_columns = [1]
data_features = data.drop(columns=data.iloc[:, label_columns])
data_features = data_features.head(100000)

data_labels = data
data_labels.columns = ['col1', 'col2']
data_labels = data_labels.head(100000)
data_labels = data_labels.astype(str)
data_labels = data_labels.apply(lambda x: strfix(x['col1'], x['col2']), axis=1)

data_features = data_features.apply(boardstate, axis=1)
data_features = data_features.apply(pd.Series)

input2_columns = [0, 1, 2, 3, 4, 5]
inputboard = data_features.drop(columns=data_features.iloc[:, input2_columns])

inputboard = np.array(inputboard)

inputmeta = data_features.iloc[:, input2_columns]
inputmeta = np.array(inputmeta)


input1 = tf.keras.layers.Input(shape=(64,))
shape1 = tf.keras.layers.Reshape(target_shape=(8, 8, 1))(input1)
conv1 = tf.keras.layers.Conv2D(kernel_size=(8, 8), padding="same", activation="relu", filters=64, input_shape=(8, 8, 1))(shape1)
bn1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-05)(conv1)
conv2 = tf.keras.layers.Conv2D(kernel_size=(8, 8), padding="same", activation="relu", filters=64, input_shape=(8, 8, 1))(bn1)
bn2 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-05)(conv2)
flatten1 = tf.keras.layers.Flatten()(bn2)
input2 = tf.keras.layers.Input(shape=(6,))

conc = tf.keras.layers.concatenate([flatten1, input2])

Denselayer1 = tf.keras.layers.Dense(1024, activation='relu')(conc)
Denselayer2 = tf.keras.layers.Dense(512, activation='relu')(Denselayer1)
Denselayer3 = tf.keras.layers.Dense(256, activation='relu')(Denselayer2)
Denselayer4 = tf.keras.layers.Dense(256, activation='relu')(Denselayer3)
Output = tf.keras.layers.Dense(1, activation='linear')(Denselayer4)

data_model = tf.keras.models.Model(inputs=[input1, input2], outputs=Output)

predictions = data_model([(inputboard[:1]), (inputmeta[:1])]).numpy

metric = [tf.keras.metrics.MeanAbsoluteError()]

opt = tf.keras.optimizers.Adam()

los = tf.keras.losses.MeanSquaredError()

data_model.compile(optimizer=opt, loss=los, metrics=metric)
data_model.summary()
data_model.fit([inputboard, inputmeta], data_labels, epochs=1000, batch_size=8192, shuffle=True)

data_model.save("engine01.keras")



