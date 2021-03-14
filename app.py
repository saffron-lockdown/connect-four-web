import io
import os
import uuid

import numpy as np
from flask import Flask, jsonify, make_response, render_template, request
from game import Game

app = Flask(__name__)

if os.getenv("ENV") == "prod":
    from model_lite import ModelLite  # For Prod

    app.model = ModelLite("models/model.tflite")  # For Prod
else:
    from model import Model  # For dev

    app.model = Model("models/m6by7-2.model")  # For dev

app.debug = False
app._static_folder = os.path.abspath("templates/static/")
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


@app.route("/", methods=["GET"])
def index():
    title = "Let's play connect 3 :D"
    resp = make_response(render_template("layouts/index.html", title=title))
    resp.set_cookie("cookie", str(uuid.uuid1()))
    return resp


@app.route("/postmethod/restart", methods=["POST"])
def restart():
    cookie = request.cookies.get("cookie")
    del_game_file(cookie)
    return {}


@app.route("/postmethod", methods=["POST"])
def post_move():

    cookie = request.cookies.get("cookie")

    move = int(request.form["move"])

    game = Game(verbose=False)
    previous_moves = read_game_file(cookie)

    for m in previous_moves:
        game.move(m)

    winner, board = game.move(move)

    if winner is not None:
        del_game_file(cookie)
        return encode(winner, board, board)

    previous_moves.append(move)

    opponent_move = get_opponent_move(board)
    winner, new_board = game.move(opponent_move)

    if winner is not None:
        del_game_file(cookie)
        return encode(winner, board, new_board)

    previous_moves.append(opponent_move)
    write_game_file(previous_moves, cookie)

    return encode(winner, board, new_board)


def get_opponent_move(board):
    return app.model.move(board, as_player=1)


def encode(winner, board, new_board):

    return jsonify({"board": board, "new_board": new_board, "winner": winner})


def read_game_file(cookie):
    try:
        with open(f"gamefiles/gamefile_{cookie}", "r") as file:
            return [int(x) for x in file.read()]
    except:
        return []


def write_game_file(moves, cookie):
    with open(f"gamefiles/gamefile_{cookie}", "w") as file:
        file.write("".join([str(x) for x in moves]))


def del_game_file(cookie):
    try:
        os.remove(f"gamefiles/gamefile_{cookie}")
    except FileNotFoundError as e:
        print(f"gamefile already deleted\n {e}")


if __name__ == "__main__":
    app.run(port=5000)
