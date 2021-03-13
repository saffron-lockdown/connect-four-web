import numpy as np
import tflite_runtime.interpreter as tflite
from game import BOARD_SIZE
from copy import deepcopy
from scipy.special import softmax
import random

class ModelLite():
    def __init__(self, load_model_name):
        # Load the TFLite model and allocate tensors.
        self.interpreter = tflite.Interpreter(model_path="models/model.tflite")
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

        return input_vector

    def board_to_vec(self, board, length=BOARD_SIZE):
        copy = deepcopy(board)
        for b in copy:
            b += [None] * (length - len(b))

        input_vec = []

        for col in copy:
            for item in col:
                if item is None:
                    input_vec += [0, 0]
                else:
                    if item == 0:
                        input_vec += [0, 1]
                    elif item == 1:
                        input_vec += [1, 0]
                    else:
                        raise Exception

        return np.array([input_vec])

    def move(self, board, as_player, print_probs=False):

        pred = self.predict(board, as_player)

        if print_probs:
            print([round(x, 2) for x in pred[0]])

        smax = softmax([x/100 for x in pred[0]])
        move = random.choices(range(len(smax)), smax)[0]

        return move

    def predict(self, board, as_player):

        # Test the model on random input data.
        # input_shape = input_details[0]['shape']
        # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], self.input_encoding(board, as_player))

        self.interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        return self.interpreter.get_tensor(self.output_details[0]['index'])

        