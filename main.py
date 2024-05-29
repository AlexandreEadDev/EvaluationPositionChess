from flask import Flask, request, jsonify
from flask_cors import CORS
import chess
import chess.engine
import random
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/evaluate', methods=['POST'])
def evaluate():
    fen = request.json['fen']
    level = request.json['level']
    board = chess.Board(fen)

    # Use the correct path to the Stockfish executable
    stockfish_path = os.environ.get('STOCKFISH_PATH', 'stockfish-ubuntu-x86-64-avx2')

    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        result = engine.analyse(board, chess.engine.Limit(time=0.1), multipv=5)
        moves_scores = [(entry['pv'][0].uci(), entry['score'].white().score(mate_score=10000)) for entry in result]

        if level == 1:
            chosen_move = random.choice(moves_scores[:5])[0]  # Chooses randomly from the top 5 moves
        elif level == 2:
            chosen_move = random.choice(moves_scores[:3])[0]  # Chooses randomly from the top 3 moves
        else:  # level == 3 or any other value
            chosen_move = moves_scores[0][0]  # Always chooses the best move

        evaluation = moves_scores[0][1]  # Evaluation of the best move

    return jsonify({'evaluation': evaluation, 'best_move': chosen_move})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
