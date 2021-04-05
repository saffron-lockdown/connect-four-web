import numpy as np
import tflite_runtime.interpreter as tflite
from game import BOARD_WIDTH, BOARD_HEIGHT
from copy import deepcopy
from scipy.special import softmax
import random


class ModelLite:
    def __init__(self, load_model_name):
        # Load the TFLite model and allocate tensors.
        self.interpreter = tflite.Interpreter(model_path="models/" + load_model_name)
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def input_encoding(self, board, as_player):

        if as_player == 1:
            input_vector = self.board_to_vec(board).astype(np.float32)
        else:
            reversed_board = [[1 - cell for cell in col] for col in board]
            input_vector = self.board_to_vec(reversed_board).astype(np.float32)

        return np.array([input_vector])

    def board_to_vec(self, board, length=BOARD_HEIGHT):
        copy = deepcopy(board)
        for b in copy:
            b += [None] * (length - len(b))

        input_layer_0 = [tile_encoding(tile) for col in copy for tile in col]

        return np.array([input_layer_0])

    def move(self, board, as_player, print_probs=False, valid_moves_only=False):
        pred = self.predict(board, as_player)

        if valid_moves_only:
            base_smax = [x / 20 for x in pred[0]]
            for i in range(BOARD_WIDTH):
                if len(board[i]) >= BOARD_HEIGHT:
                    base_smax[i] = -9999
            smax = softmax(base_smax)
        else:
            smax = softmax([x / 20 for x in pred[0]])
        
        if print_probs:
            print([round(x, 2) for x in pred[0]])
            print([round(x, 2) for x in smax])

        move = random.choices(range(len(smax)), smax)[0]
        return move
     
    def predict(self, board, as_player):

        # Test the model on random input data.
        # input_shape = input_details[0]['shape']
        # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        print(self.input_details)
        print(self.input_details[0]["index"])
        print(self.input_encoding(board, as_player))
        print(self.input_encoding(board, as_player).shape)

        self.interpreter.set_tensor(
            self.input_details[0]["index"], self.input_encoding(board, as_player)
        )

        self.interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        return self.interpreter.get_tensor(self.output_details[0]["index"])

def tile_encoding(x):
    if x == 0:
        return 1
    elif x == 1:
        return -1
    else:
        return 0