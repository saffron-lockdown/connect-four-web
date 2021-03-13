import datetime
import math
import os
import random
import time
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm import tqdm

from game import BOARD_WIDTH, BOARD_HEIGHT, Game, play_game, run_game
from model import BasicModel, Me, Model, RandomModel
from modified_tb import ModifiedTensorBoard
from tensorflow import keras
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.models import Sequential

# Tensorflow setting
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def training_loop(training_model, opponent_model, verbose=False):

    winner = None

    # for tensor board logging
    log_dir = (
        "logs/fit/"
        + training_model._name
        + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    tensorboard = ModifiedTensorBoard(log_dir=log_dir)

    # now execute the q learning
    y = 0.9
    eps = 0.5
    interval_size = 5
    num_episodes = interval_size * 1
    decay_factor = (1000 * eps) ** (
        -1 / num_episodes
    )  # ensures that eps = 0.001 after `num_episodes` episodes

    r_avg_list = []
    sse_avg_list = []
    wins = []
    n_moves_list = []
    moves_played = [0] * BOARD_WIDTH

    
    for i in tqdm(range(num_episodes), desc="Training"):
        
        avg_reward = sum(r_avg_list[-100:])/100
        if avg_reward > 950:  # TODO: implement early stopping if avg reward is already high - 
            print(f"Stopping early, average reward has reached {round(avg_reward,2)}")
            break

        as_player = random.choice([0, 1])
        eps *= decay_factor

        g = Game(verbose=verbose)

        if as_player == 1:  # Training as player 1 so opponent makes first move
            winner, board = g.move(opponent_model.move(g._board, 0))
        else:
            board = g._board

        done = False
        r_sum = 0
        move_num = 0
        while not done:
            random_move = False
            move_num += 1
            preds = training_model.predict(board, as_player)

            # To encourage early exploration
            if np.random.random() < eps:
                move = np.random.randint(0, BOARD_WIDTH - 1)
                random_move = True
            else:
                move = training_model.move(board, as_player)
                moves_played.append(move)

            winner, new_board = g.move(move)

            if winner is None:
                opponent_move = opponent_model.move(new_board, 1 - as_player)
                winner, new_board = g.move(opponent_move)

            # Calculate reward amount
            if winner == as_player:
                done = True
                wins.append(1)
                r = 1000 - move_num ** 2
            elif winner == 1 - as_player:
                done = True
                wins.append(0)
                r = -(1000 - move_num ** 2)
            elif winner == -1:
                done = True
                wins.append(None)
                r = 1000
            else:
                r = move_num

            if winner is None:
                target = r + y * np.max(training_model.predict(new_board, as_player))
            else:
                target = r

            target_vec = deepcopy(preds[0])
            target_vec[move] = target

            training_model.fit_one(
                as_player,
                board,
                np.array([target_vec]),
                epochs=1,
                verbose=0,
                callbacks=[tensorboard],
            )

            new_preds = training_model.predict(board, as_player)

            sse = sum([(x - y) ** 2 for x, y in zip(preds[0], target_vec)])
            new_sse = sum([(x - y) ** 2 for x, y in zip(new_preds[0], target_vec)])

            if verbose:
                print(
                    f"""
                    {training_model._name} training as player: {as_player}, move: {move_num}, eps: {round(eps, 2)},
                    old preds: {[round(p, 2) for p in preds[0]]}, rand move: {random_move},
                    tgt preds: {[round(p, 2) for p in target_vec]}, reward: {r},
                    new preds: {[round(p, 2) for p in new_preds[0]]}, average last 20 games: {round(sum(r_avg_list[-20:])/20, 2)} 
                    sse: {round(sse, 4)} >> {round(new_sse, 4)}
                    """
                )

            board = new_board
            r_sum += r

        if verbose and ((i % interval_size == 0 and i > 0) or (i == num_episodes - 1)):
            run_game(training_model, opponent_model, verbose=True)

        # Collect game level metrics
        r_avg_list.append(round(r_sum, 2))
        n_moves_list.append(move_num)

        tensorboard.update_stats(
            reward_sum=r_sum, wins=wins[-1], n_moves_avg=n_moves_list[-1]
        )
        tensorboard.update_dist(moves_played=moves_played)


def performance_stats(model1, model2, verbose=False, N_RUNS=50):

    wins = loss = draw = 0
    model1._moves = []

    for _ in tqdm(range(N_RUNS), desc="Scoring"):
        result = run_game(alg0=model1, alg1=model2, verbose=False)
        if result == 0:
            wins += 1
        elif result == 1:
            loss += 1
        else:
            draw += 1

    p = wins / N_RUNS
    ci = 1.96 * p * (1 - p) / math.sqrt(N_RUNS)
    print(
        f"As player 0: wins/draws/losses = {100*wins/N_RUNS}/{100*draw/N_RUNS}/{100*loss/N_RUNS}% +/={round(100*ci,1)}%"
    )

    print("moves played:")
    print(pd.Series(model1._moves).value_counts(normalize=True).sort_index())

    win_rate = wins
    wins = loss = draw = 0
    model1._moves = []

    for _ in tqdm(range(N_RUNS), desc="Scoring"):
        result = run_game(alg0=model2, alg1=model1, verbose=False)
        if result == 1:
            wins += 1
        elif result == 0:
            loss += 1
        else:
            draw += 1

    p = wins / N_RUNS
    ci = 1.96 * p * (1 - p) / math.sqrt(N_RUNS)
    print(
        f"As player 1: wins/draws/losses = {100*wins/N_RUNS}/{100*draw/N_RUNS}/{100*loss/N_RUNS}% +/={round(100*ci,1)}%"
    )
    print("moves played:")
    print(pd.Series(model1._moves).value_counts(normalize=True).sort_index())

    return (1.0 * (win_rate + wins)) / (2 * N_RUNS)


basic = BasicModel()
random_model = RandomModel()

m1 = Model(model_name="mtest1.model")
m2 = Model(model_name="mtest2.model")

log_dir = "logs/overall/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_overall = ModifiedTensorBoard(log_dir=log_dir)


for i in range(2):
    print(f"\n\n\nround {i}\n\n\n")
    if i > 0:
        time.sleep(1)

    training_loop(m1, m2, verbose=True)
    training_loop(m2, m1, verbose=True)

    tensorboard_overall.update_stats(
        m1_win_rate_v_basic=performance_stats(m1, basic),
        m2_win_rate_v_basic=performance_stats(m2, basic),
        m1_win_rate_v_random=performance_stats(m1, random_model),
        m2_win_rate_v_random=performance_stats(m2, random_model),
    )

    m1.save()
    m2.save()


# run_game(training_model, basic_model, verbose=True)
# for _ in range(3):
#     play_game(m1)
#     play_game(m2)
