import numpy as np
from common import initialize_game_state, check_end_state, check4, apply_player_action, PLAYER1, PLAYER2, NO_PLAYER, pretty_print_board, \
    connected_four
from agent_minimax.minimax import minimax

board = np.array([[0, 1, 2, 1, 2, 1, 2],
                  [0, 1, 1, 0, 0, 0, 0],
                  [0, 2, 0, 1, 0, 0, 0],
                  [0, 2, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0]])
#board = np.array([[1, 2, 1, 2, 1, 2, 0], [0, 1, 2, 1, 0, 2, 0], [0, 0, 1, 2, 0, 2, 0], [0, 0, 0, 0, 0, 0, 0],
                   #   [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]])


#apply_player_action(board, 3, PLAYER1)
#apply_player_action(board, 4, PLAYER2)
#f = arr = np.arange(6)
#print(f)
print(pretty_print_board(board))
x,y = minimax(board, PLAYER1, 4)
#print('result = ',x,' , ',y)

#x = agent_hint(board, PLAYER2, 3)
print('result = ',x)

#print(pretty_print_board(board))
#b = pick_best_move(board, PLAYER2)
#print('b = ',b)


#print('BestM;ove = ',best_move(board, PLAYER1))

"""
def minimax(board: np.ndarray, player: BoardPiece, depth: int):
    print('\n************* DEPTH = ',depth,'*************\n')
    if depth == 0 or connected_four(board, player, None):
        print('TRIUEEE')
        return best_move(board, player)

    else:
        print('Ausgangspunkt:\n',pretty_print_board(board))
        maxEval = -1000
        for move in range(0, 7):
            if board[5, move] == NO_PLAYER:
                temp_board = apply_player_action(board, move, player, True)
                print('\n ',move,'. TempBoard =',pretty_print_board(temp_board))
                eval = -minimax(temp_board, other_player(player), depth - 1)
                maxEval = max(maxEval, eval)
                print('\n ', move,'. Eval = ', eval,' & maxEval = ',maxEval)

    return maxEval
"""