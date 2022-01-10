from enum import Enum
from typing import Optional
import numpy as np
from typing import Callable, Tuple

BoardPiece = np.int8  # The data type (dtype) of the board
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 (player to move first) has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 (player to move second) has a piece

BoardPiecePrint = str  # dtype for string representation of BoardPiece
NO_PLAYER_PRINT = BoardPiecePrint(' ')
PLAYER1_PRINT = BoardPiecePrint('X')
PLAYER2_PRINT = BoardPiecePrint('O')
PlayerAction = np.int8  # The column to be played
CONNECT_N = np.int8


class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0


class SavedState:
    pass


GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    Tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]


def other_(player: BoardPiece) -> BoardPiece:
    if player == None:
        return PLAYER1
    return BoardPiece(player % 2 + 1)


def col_not_full(board: np.ndarray, col: int) -> bool:
    # print('Col', col, ' not occupied?...')
    return board[5, col] == NO_PLAYER


'''def generate_moves(board: np.ndarray, player: BoardPiece):
    # define states list (move list - list of available actions to consider)
    moves = []

    # loop over board columns
    for col in range(7):
        # make sure that current column is not full
        if board[5, col] == NO_PLAYER:
            moves.insert(col, apply_player_action(board, col, player, True))
        # else:
        # moves[col] = (None, None)

    # return the list of available moves (board class instances)
    return moves'''


def generate_states(board: np.ndarray, player: BoardPiece):
    # define states list (move list - list of available actions to consider)
    states = {}
    # loop over board columns
    for col in range(7):
        # make sure that current column is not full
        if board[5, col] == NO_PLAYER:
            states[str(col)] = apply_player_action(board, col, player, True)

    # return the list of available states (moves, board class instances)
    return states


def initialize_game_state() -> np.ndarray:
    return np.full((6, 7), NO_PLAYER, BoardPiece)


def tos(x) -> str:
    if x == NO_PLAYER:
        return NO_PLAYER_PRINT
    elif x == PLAYER1:
        return PLAYER1_PRINT
    elif x == PLAYER2:
        return PLAYER2_PRINT


def toarr(x) -> BoardPiece:
    if x == 'X':
        return PLAYER1
    elif x == 'O':
        return PLAYER2
    else:
        return NO_PLAYER


class ColumnFullException(Exception):
    """Raised when a column is already fully loaded with BoardPieces"""
    pass


def pretty_print_board(board: np.ndarray) -> str:
    s = "|==============|\n"
    for i in range(5, -1, -1):
        s = s + '|'
        for j in range(0, 7, 1):
            s = s + tos(board[i, j]) + ' '  # concatenate
        s = s + '|\n'

    s = s + "|==============|\n|0 1 2 3 4 5 6 |"
    return s

    """
    Should return `board` converted to a human readable string representation,
    to be used when playing or printing diagnostics to the console (stdout). The piece in
    board[0, 0] should appear in the lower-left. Here's an example output, note that we use
    PLAYER1_Print to represent PLAYER1 and PLAYER2_Print to represent PLAYER2):
    |==============|
    |              |
    |              |
    |              |
    |        X X   |
    |  X 0 O O 0 X |
    |  X O 0 X   X |
    |==============|
    |0 1 2 3 4 5 6 |
    """


def string_to_board(pp_board: str) -> np.ndarray:
    """
    Takes the output of pretty_print_board and turns it back into an ndarray.
    This is quite useful for debugging, when the agent crashed and you have the last
    board state as a string.
    """
    s_ind = 15
    arr = np.full((6, 7), NO_PLAYER, BoardPiece)
    for i in range(5, -1, -1):
        s_ind = s_ind + 3  # to over jump the characters | \n |
        for j in range(0, 7, 1):
            arr[i, j] = toarr(pp_board[s_ind])  # char in zahl umgewandelt ins array eintragen
            s_ind = s_ind + 2  # as the BoardPiece-entrys are seperatet by a space
    return arr


def apply_player_action(
        board: np.ndarray, action: PlayerAction, player: BoardPiece, copy: bool = False
) -> np.ndarray:
    """
    Sets board[i, action] = player, where i is the lowest open row. The modified
    board is returned. If copy is True, makes a copy of the board before modifying it.
    """
    if copy:
        board_copy = board.copy()
        for i in range(0, 6):
            if board_copy[i, action] == NO_PLAYER:
                board_copy[i, action] = player
                return board_copy
    else:
        for i in range(0, 6):
            if board[i, action] == NO_PLAYER:
                board[i, action] = player
                return board
    raise ColumnFullException("Column full! Choose different action!")


def check4(sl: np.ndarray, player: BoardPiece) -> bool:  #
    if len(sl) < 4:
        return False
    else:
        count = 0
        for i in range(len(sl)):
            # compare adjacent fields and count
            if sl[i] == player:
                count += 1
            else:
                count = 0
            if count == 4:
                return True
    return False


def get_row(board: np.ndarray, last_action: int) -> int:
    for row in range(1, 6):
        if board[row, last_action] == NO_PLAYER:
            return row - 1
    return 5


def count(slice: np.ndarray, player: BoardPiece):
    count = 1
    longest = 1
    for piece in range(1, len(slice)):
        if slice[piece - 1] == slice[piece] == player:
            count += 1
            if count > longest:
                longest = count
        else:
            count = 1
    return longest


def connected_four(
        board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> bool:
    """
      Returns True if there are four adjacent pieces equal to `player` arranged
      in either a horizontal, vertical, or diagonal line. Returns False otherwise.
      """
    # separate
    if last_action is not None:
        last_action_row = get_row(board, last_action)

        if count(board[:, last_action], player) == 4:
            return True
        if count(board[last_action_row, :], player) == 4:
            return True
        d_up = board.diagonal(last_action - last_action_row)
        if count(d_up, player) == 4:
            return True
        d_down = np.fliplr(board).diagonal(6 - (last_action_row + last_action))
        if count(d_down, player) == 4:
            return True
    else:
        for x in range(-5, 7):
            # capture ascending diagonal
            d_up = np.fliplr(board).diagonal(x)
            # capture falling diagonal as slice
            d_down = board.diagonal(x)
            if check4(d_up, player):
                # print('d_up for x = ',x)
                return True
            if check4(d_down, player):
                # print('d_down for x = ',x)
                return True
        for a in range(0, 7):
            if check4(board[:, a], player):
                # print('board[:,a] for a = ',a)
                return True
            if a < 6:
                if check4(board[a, :], player):
                    # print('board[a,:] for a = ',a)
                    return True
    return False

    ''' if check4(d_up, player) or check4(d_down, player) or check4(board[:, a], player):
                return True

    
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.
    '''


def check_end_state(board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
                    ) -> GameState:
    # did last_action lead to victory?

    if connected_four(board, player, last_action):
        return GameState(1)
    else:
        # ..still free fields left? => GameState(0)
        for i in range(0, 7):
            if col_not_full(board, i):
                return GameState(0)
        return GameState(-1)
    # ..No Field left => is draw


def game_end(board: np.ndarray) -> bool:
    if check_end_state(board, PLAYER1) != GameState(0) or check_end_state(board, PLAYER2) != GameState(0):
        return True
    return False


def winner(board: np.ndarray) -> BoardPiece:
    if connected_four(board, PLAYER1):
        return PLAYER1
    elif connected_four(board, PLAYER2):
        return PLAYER2
    else:
        return NO_PLAYER


def is_draw(board: np.ndarray) -> bool:
    return NO_PLAYER not in board[5, :]


def game_end_through_last_action(board: np.ndarray, last_player: BoardPiece,
                                 last_action: Optional[PlayerAction] = None) -> bool:
    if connected_four(board, last_player, last_action) or is_draw(board):
        return True
    return False
