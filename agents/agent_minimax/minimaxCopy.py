import random

import numpy as np
from typing import Optional
from typing import Tuple
from agents.common import NO_PLAYER, game_end, PLAYER1, PLAYER2, tos, PLAYER1_PRINT, PLAYER2_PRINT, GameState, \
    check_end_state, apply_player_action, BoardPiece, \
    connected_four, pretty_print_board, SavedState, PlayerAction


COL_HEIGHT = 7
ROW_LEN = 6

disable_jit = True
if disable_jit:
    import os
    os.environ['NUMBA_DISABLE_JIT'] = '1'

import timeit
import numpy as np
from numba import njit
from scipy.signal.sigtools import _convolve2d
from agents.common import connected_four, initialize_game_state, BoardPiece, PlayerAction, NO_PLAYER, CONNECT_N

@njit()
def connected_four_iter(
    board: np.ndarray, player: BoardPiece, _last_action: Optional[PlayerAction] = None
) -> bool:
    rows, cols = board.shape
    rows_edge = rows - CONNECT_N + 1
    cols_edge = cols - CONNECT_N + 1
    # board [Row,Col] ;-)
    for i in range(rows):
        for j in range(cols_edge):
            if np.all(board[i, j:j+CONNECT_N] == player): #horizontal
                return True

    for i in range(rows_edge):
        for j in range(cols):
            if np.all(board[i:i+CONNECT_N, j] == player): #vertical
                return True

    for i in range(rows_edge):
        for j in range(cols_edge):
            block = board[i:i+CONNECT_N, j:j+CONNECT_N] #diagonal up
            if np.all(np.diag(block) == player):
                return True
            if np.all(np.diag(block[::-1, :]) == player):
                return True

    return False


col_kernel = np.ones((CONNECT_N, 1), dtype=BoardPiece)
row_kernel = np.ones((1, CONNECT_N), dtype=BoardPiece)
dia_l_kernel = np.diag(np.ones(CONNECT_N, dtype=BoardPiece))
dia_r_kernel = np.array(np.diag(np.ones(CONNECT_N, dtype=BoardPiece))[::-1, :])


def connected_four_convolve(
    board: np.ndarray, player: BoardPiece, _last_action: Optional[PlayerAction] = None
) -> bool:
    board = board.copy()

    other_player = BoardPiece(player % 2 + 1)
    board[board == other_player] = NO_PLAYER
    board[board == player] = BoardPiece(1)

    for kernel in (col_kernel, row_kernel, dia_l_kernel, dia_r_kernel):
        result = _convolve2d(board, kernel, 1, 0, 0, BoardPiece(0))
        if np.any(result == CONNECT_N):
            return True
    return False


board = initialize_game_state()

number = 10**4

res = timeit.timeit("connected_four_iter(board, player)",
                    setup="connected_four_iter(board, player)",
                    number=number,
                    globals=dict(connected_four_iter=connected_four_iter,
                                 board=board,
                                 player=BoardPiece(1)))

print(f"Python iteration-based: {res/number*1e6 : .1f} us per call")

res = timeit.timeit("connected_four_convolve(board, player)",
                    number=number,
                    globals=dict(connected_four_convolve=connected_four_convolve,
                                 board=board,
                                 player=BoardPiece(1)))

print(f"Convolve2d-based: {res/number*1e6 : .1f} us per call")

res = timeit.timeit("connected_four(board, player)",
                    setup="connected_four(board, player)",
                    number=number,
                    globals=dict(connected_four=connected_four,
                                 board=board,
                                 player=BoardPiece(1)))

print(f"My secret sauce: {res/number*1e6 : .1f} us per call")
If I run the above code, I get the output:
Python iteration-based:  419.0 us per call
Convolve2d-based:  44.0 us per call
My secret sauce: <this should be your implementation's run-time>


def col_not_full(board: np.ndarray, col: int) -> bool:
    # print('Col', col, ' not occupied?...')
    return board[5, col] == NO_PLAYER


def other(player: BoardPiece) -> BoardPiece:
    if player == PLAYER1:
        return PLAYER2
    else:
        return PLAYER1


def get_score(board: np.ndarray, player: BoardPiece) -> int:
    # examines and evaluates all possible four-sections according
    # to the following assignments:
    # 4 PlayerPieces := adding 1000 to score
    # 3 PlayerPieces and 1 NoPlayerPiece := adding 100 to score
    # 2 PlayerPieces and 2 NoPlayerPiece := adding 10 to score
    score = 0
    bol_board = np.zeros((6, 7), dtype=bool)
    for col in range(0, 7):
        for row in range(0, 6):
            if board[row, col] != NO_PLAYER:
                bol_board[row, col] = True
            if board[row, col] == NO_PLAYER:
                bol_board[row, col] = True
                break
    # horizontal
    for row in range(0, ROW_LEN):
        single_row = board[row, :]
        bsingle_row = bol_board[row, :]
        for i in range(len(single_row) - 3):
            section = single_row[i:i + 4]
            if count_similar_pieces(section, player) == 4:
                score += 4000
            elif count_similar_pieces(section, player) == 3 and count_similar_pieces(section, NO_PLAYER) == 1:
                if bsingle_row[i:i + 4].all():
                    score += 1000
                else:
                    score += 500
            elif count_similar_pieces(section, player) == 2 and count_similar_pieces(section, NO_PLAYER) == 2:
                score += 10

    # vertical
    for col in range(0, COL_HEIGHT):
        single_col = board[:, col]
        bsingle_col = bol_board[:, col]
        for j in range(0, len(single_col) - 3):
            section = single_col[j:j + 4]
            if count_similar_pieces(section, player) == 4:
                score += 4000
            elif count_similar_pieces(section, player) == 3 and count_similar_pieces(section, NO_PLAYER) == 1:
                if bsingle_col[j:j + 4].all():
                    score += 1000
                else:
                    score += 500
            elif count_similar_pieces(section, player) == 2 and count_similar_pieces(section, NO_PLAYER) == 2:
                score += 10

    # diagonal
    for x in range(-5, 7):
        # capture rising diagonal
        d_up = np.fliplr(board).diagonal(x)
        db_up = np.fliplr(bol_board).diagonal(x)
        # capture falling diagonal
        d_down = board.diagonal(x)
        db_down = bol_board.diagonal(x)
        if len(d_up > 3):
            for z in range(0, len(d_up) - 3):
                section_up = d_up[z:z + 4]  # section of 4 pieces on rising diagonal
                section_down = d_down[z:z + 4]  # section of 4 pieces on falling diagonal
                # print('x=',x,',z=',z,'\n',section_up,'\n',section_down,'\n\n')
                if count_similar_pieces(section_up, player) == 4:
                    score += 4000
                elif count_similar_pieces(section_up, player) == 3 and count_similar_pieces(section_up, NO_PLAYER) == 1:
                    if db_up[z:z + 4].all():
                        score += 4000
                    else:
                        score += 500
                elif count_similar_pieces(section_up, player) == 2 and count_similar_pieces(section_up, NO_PLAYER) == 2:
                    score += 10
                if count_similar_pieces(section_down, player) == 4:
                    score += 4000
                elif count_similar_pieces(section_down, player) == 3 and count_similar_pieces(section_down,NO_PLAYER) == 1:
                    if db_down[z:z + 4].all():
                        score += 4000
                    else:
                        score += 500
                elif count_similar_pieces(section_down, player) == 2 and count_similar_pieces(section_down, NO_PLAYER) == 2:
                    score += 10
    return score

def count_similar_pieces(four_section: np.ndarray, player: BoardPiece) -> int:
    # counts, how many pieces of player can be find in this four_section
    count = 0
    if len(four_section) < 4:
        return 0
    # print('section = ',four_section)
    for each in four_section:
        if each == player:
            count += 1
    return count


def game_over(board: np.ndarray) -> bool:
    if connected_four(board, PLAYER1) or connected_four(board, PLAYER2):
        return True
    if board[5,:].all() != NO_PLAYER: #IS_DRAW
        return True
    return False

def calc_score(board: np.ndarray,  player: BoardPiece):
    player_score = get_score(board, player)
    enemy_score = get_score(board, other(player))
    return (player_score - enemy_score)


def minimax(board: np.ndarray, player: BoardPiece, depth: int, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    high_score = float('-inf')
    best_col = random.randint(0, 7)
    score = float('-inf')
    for col in range(0, 7):
        pos_move = board.copy()
        if col_not_full(board,col):
            apply_player_action(pos_move, col, player) # possible move
            score = max(score, -negamax(pos_move, other(player), depth, float('-inf'), float('+inf')))
            if score > high_score:
                high_score = score
                best_col = col
    return best_col, saved_state



def negamax(board: np.ndarray,  player: BoardPiece, depth: int, alpha: float, beta: float) -> int :

    if depth == 0 or game_over(board):
        return calc_score(board,player)

    else:
        value = float('-inf')
        for col in range(0,7):
            temp_board = board.copy()
            if col_not_full(board, col):
                apply_player_action(temp_board, col, player)
                value = max(value, -negamax(temp_board, other(player), depth-1, -beta,-alpha))
                alpha = max(value, alpha)
                if alpha >= beta:
                    break
        return value




