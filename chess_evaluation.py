from flask import Flask, request, jsonify
import tensorflow as tf
import chess
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("engine01.keras")

def boardstate(fen):
    board = chess.Board(fen)
    fstr = str(fen)

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

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    fen = data['fen']
    new_data_features = [boardstate(fen)]
    new_data_features_df = pd.DataFrame(new_data_features)
    input2_columns = [0, 1, 2, 3, 4, 5]
    new_inputboard = new_data_features_df.drop(columns=new_data_features_df.iloc[:, input2_columns])
    new_inputmeta = new_data_features_df.iloc[:, input2_columns]
    new_inputboard = np.array(new_inputboard)
    new_inputmeta = np.array(new_inputmeta)

    prediction = model.predict([new_inputboard, new_inputmeta])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
