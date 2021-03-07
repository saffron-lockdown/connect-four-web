import glob
import io
import os
import uuid

import numpy as np
from flask import Flask, jsonify, make_response, render_template, request
from game import Game

app = Flask(__name__)
app.secret_key = "s3cr3t"
app.debug = False
app._static_folder = os.path.abspath("templates/static/")

@app.route("/", methods=["GET"])
def index():
    title = "Let's play connect 3 :D"
    return render_template("layouts/index.html", title=title)

@app.route("/postmethod", methods=["POST"])
def post_move():
    move = int(request.form["move"])
    
    game = Game()
    previous_moves = read_game_file()
    
    for m in previous_moves:
        game.move(m)
    
    winner, board = game.move(move)
    
    if winner is not None:
        del_game_file()
        return encode(winner, board)
    
    previous_moves.append(move)

    opponent_move = 3
    winner, board = game.move(opponent_move)
    
    if winner is not None:
        del_game_file()
        return encode(winner, board)
    
    previous_moves.append(opponent_move)
    write_game_file(previous_moves)

    return encode(winner, board)

def encode(winner, board):
    
    return jsonify({
        "board": board,
        "winner": winner
    })

def read_game_file():
    try:
        with open('gamefile', "r") as file:
            return [int(x) for x in file.read()]
    except:
        return []

def write_game_file(moves):
    with open('gamefile', "w") as file:
        file.write("".join([str(x) for x in moves]))

def del_game_file():
    os.remove('gamefile')
    return

if __name__ == "__main__":
    app.run(port=5000)
