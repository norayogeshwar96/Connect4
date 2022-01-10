import pytest
import numpy as np
from agents.MCTS.monte_carlo_ts import TreeNode, uct, createnode, simulate,pick_best_move, back_propagate, expand
from agents.common import initialize_game_state, PLAYER1, PLAYER2


def test_createnode():
    board = initialize_game_state()
    treenode = createnode(board, None)
    assert isinstance(treenode, TreeNode)
    assert (treenode.parent is None)
    assert (treenode.acting_player == PLAYER1)


def test_expand():
    board = initialize_game_state()
    treenode = createnode(board, None)
    child = expand(treenode)
    assert (child.parent is treenode)
    assert (child.last_action in treenode.children.keys())
    child2 = expand(treenode)
    child3 = expand(treenode)
    child4 = expand(treenode)
    child5 = expand(treenode)
    child6 = expand(treenode)
    child7 = expand(treenode)
    child8 = expand(treenode)
    assert isinstance(child7, TreeNode)
    assert (child8 is None)
    assert (child2.last_action in treenode.children.keys())
    assert (child3.last_action in treenode.children.keys())
    assert (child4.last_action in treenode.children.keys())
    assert (child5.last_action in treenode.children.keys())
    assert (child6.last_action in treenode.children.keys())
    assert (child7.last_action in treenode.children.keys())
    assert (child3.parent is treenode)

def test_simulate():
    board = np.array([[1, 1, 1, 2, 1, 1, 1],
                      [1, 2, 1, 2, 1, 1, 1],
                      [2, 2, 2, 1, 2, 2, 2],
                      [1, 2, 2, 1, 2, 2, 1],
                      [0, 0, 2, 0, 2, 2, 1],
                      [0, 0, 1, 0, 0, 0, 2]])
    t = TreeNode(board, None)

    board = np.array([[1, 1, 1, 2, 1, 1, 1],
                      [1, 2, 1, 2, 1, 1, 1],
                      [2, 2, 2, 1, 2, 2, 2],
                      [1, 2, 2, 1, 2, 2, 1],
                      [1, 0, 2, 0, 2, 2, 1],
                      [0, 0, 1, 0, 0, 0, 2]])
    tt = TreeNode(board, t)

    board = np.array([[1, 1, 1, 2, 1, 1, 1],
                      [1, 2, 1, 2, 1, 1, 1],
                      [2, 2, 2, 1, 2, 2, 2],
                      [1, 2, 2, 1, 2, 2, 1],
                      [1, 2, 2, 0, 2, 2, 1],
                      [0, 0, 1, 0, 0, 0, 2]])
    ttt = TreeNode(board, tt)

    board = np.array([[1, 1, 1, 2, 1, 1, 1],
                      [1, 2, 1, 2, 1, 1, 1],
                      [2, 2, 2, 1, 2, 2, 2],
                      [1, 2, 2, 1, 2, 2, 1],
                      [1, 2, 2, 0, 2, 2, 1],
                      [1, 0, 1, 0, 0, 0, 2]])
    node = TreeNode(board, ttt)

    assert (simulate(node) == -1)

def test_back_propagate():
    board = np.array([[1, 1, 1, 2, 1, 1, 1],
                      [1, 2, 1, 2, 1, 1, 1],
                      [2, 2, 2, 1, 2, 2, 2],
                      [1, 2, 2, 1, 2, 2, 1],
                      [0, 0, 2, 0, 2, 2, 1],
                      [0, 0, 1, 0, 0, 0, 2]])
    t = TreeNode(board, None)

    board = np.array([[1, 1, 1, 2, 1, 1, 1],
                      [1, 2, 1, 2, 1, 1, 1],
                      [2, 2, 2, 1, 2, 2, 2],
                      [1, 2, 2, 1, 2, 2, 1],
                      [1, 0, 2, 0, 2, 2, 1],
                      [0, 0, 1, 0, 0, 0, 2]])
    tt = TreeNode(board, t)

    board = np.array([[1, 1, 1, 2, 1, 1, 1],
                      [1, 2, 1, 2, 1, 1, 1],
                      [2, 2, 2, 1, 2, 2, 2],
                      [1, 2, 2, 1, 2, 2, 1],
                      [1, 2, 2, 0, 2, 2, 1],
                      [0, 0, 1, 0, 0, 0, 2]])
    ttt = TreeNode(board, tt)

    board = np.array([[1, 1, 1, 2, 1, 1, 1],
                      [1, 2, 1, 2, 1, 1, 1],
                      [2, 2, 2, 1, 2, 2, 2],
                      [1, 2, 2, 1, 2, 2, 1],
                      [1, 2, 2, 0, 2, 2, 1],
                      [1, 0, 1, 0, 0, 0, 2]])
    node = TreeNode(board, ttt)
    score = simulate(node)
    back_propagate(node, score)

    assert(t.scored_wins == -1)
    assert(t.visits == 1)

'''def uct():
    board = np.array([[2, 1, 1, 2, 1, 1, 1],
                      [0, 2, 1, 2, 1, 0, 1],
                      [0, 0, 2, 1, 2, 0, 0],
                      [0, 0, 0, 1, 2, 0, 0],
                      [0, 0, 0, 0, 2, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]])
    node = TreeNode(board,None)
    board4 = np.array([[2, 1, 1, 2, 1, 1, 1],
                      [0, 2, 1, 2, 1, 0, 1],
                      [0, 0, 2, 1, 2, 0, 0],
                      [0, 0, 0, 1, 2, 0, 0],
                      [0, 0, 0, 0, 2, 0, 0],
                      [0, 0, 0, 0, 2, 0, 0]])
    board0 = np.array([[2, 1, 1, 2, 1, 1, 1],
                       [2, 2, 1, 2, 1, 0, 1],
                       [0, 0, 2, 1, 2, 0, 0],
                       [0, 0, 0, 1, 2, 0, 0],
                       [0, 0, 0, 0, 2, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0]])
    board1 = np.array([[2, 1, 1, 2, 1, 1, 1],
                       [0, 2, 1, 2, 1, 0, 1],
                       [0, 2, 2, 1, 2, 0, 0],
                       [0, 0, 0, 1, 2, 0, 0],
                       [0, 0, 0, 0, 2, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0]])
    board2 = np.array([[2, 1, 1, 2, 1, 1, 1],
                       [0, 2, 1, 2, 1, 0, 1],
                       [0, 0, 2, 1, 2, 0, 0],
                       [0, 0, 2, 1, 2, 0, 0],
                       [0, 0, 0, 0, 2, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0]])
    
    child4 = TreeNode(board1,node)
    child0 = '''





