# main function for the Monte Carlo Tree Search
import numpy as np
import math
import random
from agents.common import game_end, BoardPiece, NO_PLAYER, connected_four, PLAYER1, is_draw, PLAYER2, winner, \
    generate_states, \
    GameState, other_, apply_player_action, game_end_through_last_action, \
    check_end_state, GameState
from agents.agent_random.random import generate_move_random

BOARD = np.ndarray


class TreeNode():
    def __init__(self, board, parent):
        self.board = board
        # the player whos turn to play it is.
        # if parent.acting_player == None => beginning of game => other_() will return PLAYER1
        if parent is None:
            self.acting_player = PLAYER1
            self.depth = 0
        else:
            self.acting_player = other_(parent.acting_player)
            self.depth = parent.depth + 1  # versuche ich mal als timesteps zu benutzen für upc (?!)
        self.parent = parent

        # no. of wins achieved out of this state, beziehen sich immer auf acting player vom parent!
        self.scored_wins = 0
        self.visits = 0
        # column played, which created this new state(node)
        # key in parents children dictionary => node.parent.children[node.last_action] = node
        self.last_action = None

        # dict of (action: node)
        self.children = {}

        if game_end(board):
            self.is_terminal = True
        else:
            self.is_terminal = False

        # if its a terminal node it is equal to fully examined
        # if its not a terminal node, fully_expanded stays False and willbe adjusted during expand-function
        self.fully_expanded = self.is_terminal


def createnode(board, parent):
    return TreeNode(board, parent)


def expand(node):
    # generates dict of legal states (moves) for the given node
    # states = {('last_action': board),('last_action': board), .... }
    if len(node.children) < 7:  # vor aufruf von expand sollte immer node.fully_expanded abgefragt werden, dann kann man das if und else hier streichen

        states = generate_states(node.board, node.acting_player)

        # loop over generated states (moves)
        for state in states:
            # make sure that current state (move) is not present in child nodes
            if str(state) not in node.children.keys():
                # create a new node
                new_node = createnode(states[state], node)
                # add last_action as attribute
                new_node.last_action = str(state)
                # add to parents children list, where node.children[str(action)] = child_node
                node.children[str(state)] = new_node

                # case when node is fully expanded
                if len(states) == len(node.children):
                    node.fully_expanded = True

                # return newly created node !!Innerhalb forschleife!!!
                return new_node

    return None

def current_player(player: BoardPiece):
    if player == PLAYER1:
        return 1
    else:
        return -1


def uct(player: int, node, child_node, c):
    return (player * child_node.scored_wins / child_node.visits + c * math.sqrt(
        math.log(node.visits / child_node.visits)))


# select the best node basing on UCB formula
def pick_best_move(node, c):  # c = exploration constant
    # define best score & best moves
    best_score = float('-inf')
    best_moves = []

    # loop over child nodes
    for child_node in node.children.values():
        # define current player out of perspective of PLAYER1
        active_player = current_player(node.acting_player)

        # get move score using UCB+MCTS formula
        move_score = uct(active_player, node, child_node, c)

        # better move has been found
        if move_score > best_score:
            best_score = move_score
            best_moves = [child_node]

        # found as good move as already available
        elif move_score == best_score:
            best_moves.append(child_node)

    # return one of the best moves randomly
    return random.choice(best_moves)


def is_terminal(node):
    return connected_four(node.board, PLAYER1) or connected_four(node.board, PLAYER2) or is_draw(node.board)


def random_move(board: np.ndarray):
    rand_order = np.arange(7)
    random.shuffle(rand_order)
    for i in rand_order:
        if board[5, i] == NO_PLAYER:
            return i
    return 0


def simulate(node):
    simulation = node.board.copy()
    player = other_(node.acting_player)
    rand_action = int(node.last_action)
    while not game_end_through_last_action(simulation, player, rand_action):
        player = other_(player)
        rand_action = random_move(simulation)
        apply_player_action(simulation, rand_action, player, False)

    # return score from the PLAYER1 perspective
    if connected_four(simulation, PLAYER1):
        return 1
    elif connected_four(simulation, PLAYER2):
        return -1

    return 0


def back_propagate(node: TreeNode, score):
    branch_node = node
    while branch_node is not None:
        branch_node.visits += 1
        branch_node.scored_wins += score
        branch_node = branch_node.parent


# search for the best move in the current position
def search(current_board: np.ndarray):
    # create root node
    root = TreeNode(current_board, None)

    # walk through 1000 iterations
    for iteration in range(1000):
        #print('i = ',iteration)
        # select a node (selection phase)
        node = select(root)

        # score current node (simulation phase)
        score = simulate(node)

        # backpropagate results
        back_propagate(node, score)

    # pick up the best move in the current position
    try:
        return pick_best_move(root, 0)

    except:
        pass


# select most promising node
def select(node):
    # make sure that we're dealing with non-terminal nodes
    while not node.is_terminal:
        # case where the node is fully expanded
        if node.fully_expanded:
            node = pick_best_move(node, 2)

        # case where the node is not fully expanded
        else:
            # otherwise expand the node
            return expand(node)

    # return node
    return node

