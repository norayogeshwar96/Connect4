def minimax(board: np.ndarray, player: BoardPiece, depth: int, alpha: int, beta: int):

    if depth == 0 or isGameWon(board,player):
        best = get_score(board, player)
        #print('PLAYER =', player,'depth ',depth,' *** ENDPUNKT *** score = ',best)
        #print(pretty_print_board(board))
        return best
    else:
        best_score = -10000
        for col in range(ROW_LEN):
            if not_occupied(board, col):
                board_copy = board.copy()
                apply_player_action(board_copy, col, player, False)
                print('Player',player,' ** depth ',depth,' Column = ',col,'\n',pretty_print_board(board_copy))
                score = -1 * (minimax(board_copy, other_player(player), depth-1, -alpha, -beta))
                if score > best_score:
                    print('Best Score for col ',col,' = ',score)
                    best_score = score
                    bestmove = col
                alpha = max(alpha, score)
                if alpha >= beta:
                    break
        print('COLUM RETURN = ',bestmove)
        return bestmove

def best_move(board: np.ndarray, player: BoardPiece):
    best_score = -1000
    for col in range(ROW_LEN):
        if not_occupied(board, col):
            board_copy = board.copy()
            apply_player_action(board_copy, col, player)
            score = get_score(board_copy, player)
            score = minimax(board, player)
            if score > best_score:
                best_move = col
                best_score = score
    return best_move
#def best_move(board: np.ndarray, player: BoardPiece, depth: int, alpha: int, beta: int)->int: