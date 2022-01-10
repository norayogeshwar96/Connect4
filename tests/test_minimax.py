import numpy as np
import pytest
from agents.common import PLAYER1, PLAYER2, NO_PLAYER,apply_player_action, pretty_print_board
from agents.agent_minimax.minimax import get_score, count_section, minimax, agent_hint


def test_count_section():
    section = np.array([1, 1, 1, 1])
    assert (count_section(section, PLAYER1) == 4)
    assert (count_section(section, PLAYER2) == 0)

    section = ([0, 1, 1, 1])
    assert (count_section(section, PLAYER1) == 3)
    section = ([0, 1, 0, 1])
    assert (count_section(section, PLAYER1) == 2)
    section = ([2, 1, 2, 1])
    assert (count_section(section, PLAYER1) == 2)
    section = ([0, 0, 0, 2])
    assert (count_section(section, PLAYER2) == 1)
    section = ([0, 0, 0, 0])
    assert (count_section(section, PLAYER2) == 0)
    assert (count_section(section, NO_PLAYER) == 4)


def test_get_score():
    board = np.array([[1, 1, 1, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]])
    assert (get_score(board, PLAYER2) == 4000)
    board[0, :] = [1, 1, 1, 0, 2, 2, 2]
    assert (get_score(board, PLAYER1) == 100)
    board = np.array([[1, 2, 2, 1, 2, 0, 0], [0, 1, 2, 2, 2, 0, 0], [0, 0, 1, 1, 2, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]])
    assert (get_score(board, PLAYER1) == 120)
    apply_player_action(board,3 ,PLAYER1)
    assert (get_score(board, PLAYER1) == 4130)

    # assert(get_score(board, PLAYER1) ==  )

def test_minimax():
    #horizontal winning move
    board = np.array([[1, 1, 1, 2, 2, 2, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]])

    assert(agent_hint(board, PLAYER2, 3) == 6)
    assert (agent_hint(board, PLAYER1, 3) == 6)

    board = np.array([[1, 2, 1, 2, 2, 2, 0], [0, 1, 2, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]])

    assert (agent_hint(board, PLAYER1, 3) == 6)

    #diagonal-up winning move
    board = np.array([[1, 2, 1, 2, 1, 2, 0], [0, 1, 2, 2, 0, 0, 0], [0, 0, 1, 2, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]])
    assert (agent_hint(board, PLAYER1, 3) == 3)
    assert (agent_hint(board, PLAYER2, 3) == 3)

    board = np.array([[0, 1, 2, 1, 0, 1, 2], [0, 1, 1, 2, 0, 0, 0], [0, 2, 2, 1, 0, 0, 0], [0, 2, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]])
    print(pretty_print_board(board))
    assert (agent_hint(board, PLAYER1, 3) == 4)
    assert (agent_hint(board, PLAYER2, 3) == 4)

    # vertikal winning move
    board = np.array([[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]])

    assert (agent_hint(board, PLAYER1, 3) == 6)