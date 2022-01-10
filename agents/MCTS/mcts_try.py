import numpy as np
from agents.common import initialize_game_state, check_end_state, check4, apply_player_action, PLAYER1, PLAYER2, NO_PLAYER, pretty_print_board, \
    connected_four
from agents.MCTS.monte_carlo_ts import search
from agents.agent_random.random import generate_move_random

board = initialize_game_state()
apply_player_action(board,4,PLAYER1)
apply_player_action(board,3,PLAYER2)
apply_player_action(board,4,PLAYER1)
apply_player_action(board,1,PLAYER2)
print(pretty_print_board(board))
move = search(board)
print('\n',pretty_print_board(move.board))
