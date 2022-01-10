from typing import Optional
from typing import Tuple
import numpy as np
from ..common import SavedState, GameState, NO_PLAYER, PlayerAction, BoardPiece
import random


def generate_move_random(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:

    rand = random.randint(0,6)
    if board[5,rand] == NO_PLAYER:
       # return rand, saved_state
         return rand
    else:
        for i in range (0,7):
            if board[5,i] == NO_PLAYER:
                #return i, saved_state
                return i

    # Choose a valid, non-full column randomly and return it as `rand`
    #return rand, saved_state
    return rand



