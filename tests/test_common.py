import numpy as np
import pytest

from agents.common import BoardPiece, check_end_state, get_row, count, check4, pretty_print_board, connected_four, \
    NO_PLAYER, PLAYER1, PLAYER2, initialize_game_state, get_row, GameState, \
    apply_player_action, ColumnFullException, generate_states, is_draw, game_end_through_last_action


def test_initialize_game_state():
    ret = initialize_game_state()

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == BoardPiece
    assert ret.shape == (6, 7)
    assert np.all(ret == NO_PLAYER)


def test_get_row():
    board = np.array([[1, 2, 1, 2, 2, 2, 1],
                      [0, 1, 2, 0, 1, 2, 1],
                      [0, 0, 1, 0, 1, 2, 2],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 0, 0, 2, 0],
                      [0, 0, 0, 0, 0, 1, 0]])
    assert (get_row(board, 6) == 2)
    assert (get_row(board, 0) == 0)
    assert (get_row(board, 1) == 1)
    assert (get_row(board, 4) == 3)
    assert (get_row(board, 5) == 5)


def test_count():
    board = np.array([[1, 2, 1, 2, 2, 2, 0],
                      [2, 1, 2, 0, 1, 0, 0],
                      [0, 2, 1, 0, 1, 0, 0],
                      [0, 0, 2, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]])
    print(pretty_print_board(board))
    assert count(board[0, :], PLAYER2) == 3
    assert count(board[0, :], PLAYER1) == 1
    assert count(board[:, 4], PLAYER1) == 3
    apply_player_action(board, 3, PLAYER1)
    last_action = 3
    last_action_row = get_row(board, 3)
    print(last_action_row, '\n')
    print(pretty_print_board(board))
    d_up = board.diagonal(last_action - last_action_row)
    print(d_up)
    assert count(d_up, PLAYER1) == 3


def test_connected_four():
    board = np.array([[1, 2, 1, 2, 2, 2, 0],
                      [0, 1, 2, 0, 1, 0, 0],
                      [0, 0, 1, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]])


def a_board() -> np.ndarray:
    board = initialize_game_state()
    board = (apply_player_action(board, 0, PLAYER1, False))
    assert board[0, 0] == PLAYER1
    board = (apply_player_action(board, 0, PLAYER2, False))
    assert board[1, 0] == PLAYER2
    board = (apply_player_action(board, 0, PLAYER1, False))
    board = (apply_player_action(board, 0, PLAYER2, False))
    board = (apply_player_action(board, 0, PLAYER1, False))

    return board


def test_generate_states():
    # key describes last action:
    board = initialize_game_state()
    board2 = apply_player_action(board, 3, PLAYER1, True)
    states = generate_states(board, PLAYER1)
    assert ((states['3'] == board2).all())

    # full columns should not be considered and though not be returned in the dict of generated states
    board[:, 5] = PLAYER2
    states2 = generate_states(board, PLAYER1)
    assert ('5' not in states2.keys())
    board = np.array([[1, 2, 1, 2, 2, 2, 1],
                      [0, 1, 2, 1, 1, 1, 2],
                      [0, 0, 1, 2, 1, 2, 1],
                      [0, 0, 0, 2, 1, 2, 1],
                      [0, 0, 0, 1, 2, 1, 2],
                      [0, 0, 0, 2, 2, 1, 2]])
    assert (len(generate_states(board, PLAYER2)) == 3)


def test_apply_player_action():
    board = initialize_game_state()

    # board: np.ndarray, action: PlayerAction, player: BoardPiece, copy: bool = False
    board = (apply_player_action(board, 0, PLAYER1, False))
    assert board[0, 0] == PLAYER1
    board = (apply_player_action(board, 0, PLAYER2, False))
    assert board[1, 0] == PLAYER2
    board = (apply_player_action(board, 0, PLAYER1, False))
    board = (apply_player_action(board, 0, PLAYER2, False))
    board = (apply_player_action(board, 0, PLAYER1, False))
    with pytest.raises(ColumnFullException):
        board = (apply_player_action(board, 0, PLAYER2, False))


def test_check_end_state():
    board = np.array([[1, 1, 2, 1, 2, 2, 2], [1, 0, 2, 2, 2, 0, 1], [1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]])
    # STILL_PLAYING
    assert (check_end_state(board, PLAYER1) == GameState(0))

    # PLAYER1 WINS, => STILL_PLAYING FOR PLAYER2
    board = apply_player_action(board, 0, PLAYER1)
    assert (check_end_state(board, PLAYER1) == GameState(1))
    assert (check_end_state(board, PLAYER2) == GameState(0))

    # IS_DRAW
    board[0, :] = [0, 1, 2, 1, 2, 2, 2]
    board[5, :] = [1, 2, 2, 1, 2, 1, 1]
    assert (check_end_state(board, PLAYER1) == GameState(-1))


def test_check4():
    arr = [0, 1, 2, 1, 2, 2, 2]
    assert check4(arr, PLAYER2) == False
    assert check4(arr, PLAYER1) == False
    arr = [0, 1, 2, 2, 2, 2, 2]
    assert check4(arr, PLAYER2) == True
    assert check4(arr, PLAYER1) == False
    arr = [0, 0, 0, 1, 1, 1, 1]
    assert check4(arr, PLAYER1) == True
    assert check4(arr, PLAYER2) == False
    arr = [1, 1, 1, 1, 0, 0, 0]
    assert check4(arr, PLAYER1) == True
    arr = [0, 0, 0, 0, 0, 0, 0]
    assert check4(arr, PLAYER2) == False
    assert check4(arr, PLAYER1) == False
    arr = [1, 1, 1]
    assert check4(arr, PLAYER2) == False
    assert check4(arr, PLAYER1) == False
    arr = [1, 1, 1, 1, 2, 2, 2]
    assert check4(arr, PLAYER1) == True


def test_connected_four():
    board = np.array([[1, 1, 1, 1, 2, 2, 2],
                      [1, 0, 2, 2, 2, 0, 1],
                      [1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]])
    assert connected_four(board, PLAYER1) == True
    assert connected_four(board, PLAYER2) == False

    board = np.array([[2, 0, 2, 1, 2, 2, 2],
                      [1, 0, 1, 2, 2, 0, 1],
                      [1, 0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]])

    assert connected_four(board, PLAYER1) == True
    assert connected_four(board, PLAYER2) == False

    board = np.array([[2, 0, 2, 1, 2, 2, 2],
                      [1, 0, 1, 2, 2, 0, 1],
                      [1, 0, 1, 0, 2, 0, 0],
                      [0, 0, 1, 0, 0, 2, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]])

    assert connected_four(board, PLAYER2) == True
    assert connected_four(board, PLAYER1) == False

    board = np.array([[2, 0, 1, 1, 2, 2, 2],
                      [1, 0, 1, 2, 1, 0, 1],
                      [1, 0, 2, 2, 2, 0, 0],
                      [0, 0, 1, 0, 2, 2, 0],
                      [0, 0, 0, 0, 0, 2, 0],
                      [0, 0, 0, 0, 0, 0, 2]])
    assert connected_four(board, PLAYER2) == True
    assert connected_four(board, PLAYER1) == False
    board = np.array([[2, 0, 1, 1, 2, 2, 2],
                      [1, 0, 1, 2, 2, 0, 1],
                      [1, 0, 2, 1, 2, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0]])
    assert connected_four(board, PLAYER1) == True
    assert connected_four(board, PLAYER2) == False

    # handing over last_action :

    board = np.array([[2, 0, 1, 1, 2, 2, 2],
                      [1, 0, 1, 2, 1, 0, 1],
                      [1, 0, 2, 1, 2, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]])
    assert connected_four(board, PLAYER1) == True
    assert connected_four(board, PLAYER1, 0) == False

    board = np.array([[2, 2, 2, 2, 0, 1, 1],
                      [1, 0, 1, 2, 1, 0, 1],
                      [1, 0, 2, 1, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]])
    assert connected_four(board, PLAYER2) == True
    assert connected_four(board, PLAYER2, 4) == False
    assert connected_four(board, PLAYER2, 1) == True
    assert connected_four(board, PLAYER1, 2) == True
    assert connected_four(board, PLAYER1, 3) == True
    assert connected_four(board, PLAYER1, 4) == True
    assert connected_four(board, PLAYER1, 5) == True

    board = np.array([[2, 1, 1, 2, 1, 1, 1],
                      [0, 2, 1, 2, 1, 0, 1],
                      [0, 0, 2, 1, 2, 0, 0],
                      [0, 0, 0, 2, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]])

    assert connected_four(board, PLAYER2, 0) == True
    assert connected_four(board, PLAYER2, 1) == True
    assert connected_four(board, PLAYER2, 2) == True
    assert connected_four(board, PLAYER2, 3) == True
    assert connected_four(board, PLAYER2, 4) == False

    board = np.array([[2, 1, 1, 2, 1, 1, 1],
                      [0, 2, 1, 2, 1, 0, 1],
                      [0, 0, 2, 1, 2, 0, 0],
                      [0, 0, 0, 1, 2, 0, 0],
                      [0, 0, 0, 0, 2, 0, 0],
                      [0, 0, 0, 0, 2, 0, 0]])

    assert connected_four(board, PLAYER2, 4) == True
    assert connected_four(board, PLAYER2, 2) == False


def test_get_row():
    board = np.array([[2, 1, 1, 2, 1, 0, 1],
                      [0, 2, 1, 2, 1, 0, 1],
                      [0, 0, 2, 1, 2, 0, 0],
                      [0, 0, 0, 1, 2, 0, 0],
                      [0, 0, 0, 1, 2, 0, 0],
                      [0, 0, 0, 0, 2, 0, 0]])
    assert (get_row(board, 2) == 2)
    assert (get_row(board,5) == 0)



def test_is_draw():
    board = np.array([[2, 1, 1, 2, 1, 1, 1],
                      [0, 2, 1, 2, 1, 0, 1],
                      [0, 0, 2, 1, 2, 0, 0],
                      [0, 0, 0, 1, 2, 0, 0],
                      [0, 0, 0, 0, 2, 0, 0],
                      [0, 0, 0, 0, 2, 0, 0]])
    assert (is_draw(board) == False)

    board = np.array([[2, 1, 1, 2, 1, 1, 1],
                      [0, 2, 1, 2, 1, 0, 1],
                      [0, 0, 2, 1, 2, 0, 0],
                      [0, 0, 0, 1, 2, 0, 0],
                      [0, 0, 0, 0, 2, 0, 0],
                      [2, 1, 1, 2, 2, 0, 2]])
    assert (is_draw(board) == False)

    board = np.array([[2, 1, 1, 2, 1, 1, 1],
                      [0, 2, 1, 2, 1, 0, 1],
                      [0, 0, 2, 1, 2, 0, 0],
                      [0, 0, 0, 1, 2, 0, 0],
                      [0, 0, 0, 0, 2, 0, 0],
                      [2, 1, 1, 2, 2, 1, 2]])

    assert (is_draw(board) == True)


def test_game_end_through_last_action():
    board = np.array([[2, 1, 1, 2, 1, 1, 1],
                      [0, 2, 1, 2, 1, 0, 1],
                      [0, 0, 2, 1, 2, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 2, 0, 0],
                      [2, 1, 1, 0, 2, 1, 2]])

    assert game_end_through_last_action(board, PLAYER1, 3) == True
    assert game_end_through_last_action(board, PLAYER1, 1) == False

    board = np.array([[2, 1, 1, 2, 1, 1, 1],
                      [0, 2, 1, 2, 1, 0, 1],
                      [0, 0, 2, 1, 2, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 2, 0, 0],
                      [2, 1, 1, 1, 2, 1, 2]])

    assert game_end_through_last_action(board, PLAYER1, 1) == True

    board = np.array([[2, 1, 1, 2, 1, 1, 1],
                      [1, 2, 1, 2, 1, 1, 1],
                      [2, 2, 2, 1, 2, 2, 2],
                      [1, 1, 2, 1, 2, 1, 2],
                      [1, 2, 1, 2, 1, 2, 1],
                      [2, 1, 1, 1, 2, 1, 2]])

    assert game_end_through_last_action(board, PLAYER2, 4) == True
    assert game_end_through_last_action(board, PLAYER2, 3) == True

    board = np.array([[2, 1, 1, 2, 1, 1, 1],
                      [1, 2, 1, 2, 1, 1, 1],
                      [2, 2, 2, 1, 2, 2, 2],
                      [1, 1, 2, 1, 2, 1, 2],
                      [1, 2, 1, 2, 1, 2, 1],
                      [2, 1, 1, 1, 2, 1, 0]])
    assert connected_four(board,PLAYER2) == True
    assert connected_four(board, PLAYER2, 4) == True

   # assert game_end_through_last_action(board, PLAYER2, 4) == True
