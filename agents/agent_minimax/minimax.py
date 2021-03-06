import random

import numpy as np
from typing import Optional
from typing import Tuple
from agents.common import NO_PLAYER, game_end, PLAYER1, PLAYER2, tos, PLAYER1_PRINT, PLAYER2_PRINT, GameState, \
    check_end_state, apply_player_action, BoardPiece, \
    connected_four, pretty_print_board, SavedState, PlayerAction, other_, col_not_full

COL_HEIGHT = 7
ROW_LEN = 6

'''Command (shift) minus/plus collapse'''



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
                elif count_similar_pieces(section_down, player) == 3 and count_similar_pieces(section_down,
                                                                                              NO_PLAYER) == 1:
                    if db_down[z:z + 4].all():
                        score += 4000
                    else:
                        score += 500
                elif count_similar_pieces(section_down, player) == 2 and count_similar_pieces(section_down,
                                                                                              NO_PLAYER) == 2:
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
    if board[5, :].all() != NO_PLAYER:  # IS_DRAW
        return True
    return False


def calc_score(board: np.ndarray, player: BoardPiece):
    player_score = get_score(board, player)
    enemy_score = get_score(board, other_(player))
    return player_score - enemy_score


def minimax(board: np.ndarray, player: BoardPiece, depth: int, saved_state: Optional[SavedState]
            ) -> Tuple[PlayerAction, Optional[SavedState]]:
    high_score = float('-inf')
    best_col = random.randint(0, 7)
    score = float('-inf')
    for col in range(0, 7):
        pos_move = board.copy()
        if col_not_full(board, col):
            apply_player_action(pos_move, col, player)  # possible move
            score = max(score, -negamax(pos_move, other_(player), depth, float('-inf'), float('+inf')))
            if score > high_score:
                high_score = score
                best_col = col
    return best_col, saved_state


def negamax(board: np.ndarray, player: BoardPiece, depth: int, alpha: float, beta: float) -> int:
    if depth == 0 or game_over(board):
        return calc_score(board, player)

    else:
        value = float('-inf')
        for col in range(0, 7):
            temp_board = board.copy()
            if col_not_full(board, col):
                apply_player_action(temp_board, col, player)
                value = max(value, -negamax(temp_board, other_(player), depth - 1, -beta, -alpha))
                alpha = max(value, alpha)
                if alpha >= beta:
                    break
        return value
