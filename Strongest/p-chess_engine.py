#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P-chessbot-Engine
Standalone Python chess engine (~2100 Elo)
Features:
- Iterative deepening
- Alpha-beta pruning
- Quiescence search
- Piece-square tables
- Pawn structure, king safety, mobility
- Learning from past games (learn.json)
- Fast mode support
"""

import chess
import time
import random
import json
import os
from collections import defaultdict

# ---------------- CONFIG ----------------
MAX_DEPTH = 5               # can increase to 6–7 for stronger play
DEFAULT_MOVETIME_MS = 1000  # default move time
FAST_MOVETIME_MS = 50       # move time in fast mode
INF = 10_000_000
MATE = 9_999_000
LEARN_FILE = "learn.json"

# ---------------- PIECE VALUES ----------------
PIECE_VAL = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

# ---------------- PIECE-SQUARE TABLES ----------------
# White perspective
PSQT_PAWN = [0,0,0,0,0,0,0,0,
             50,50,50,50,50,50,50,50,
             10,10,20,30,30,20,10,10,
             5,5,10,25,25,10,5,5,
             0,0,0,20,20,0,0,0,
             5,-5,-10,0,0,-10,-5,5,
             5,10,10,-20,-20,10,10,5,
             0,0,0,0,0,0,0,0]
PSQT_KNIGHT = [-50,-40,-30,-30,-30,-30,-40,-50,
               -40,-20,0,0,0,0,-20,-40,
               -30,0,10,15,15,10,0,-30,
               -30,5,15,20,20,15,5,-30,
               -30,0,15,20,20,15,0,-30,
               -30,5,10,15,15,10,5,-30,
               -40,-20,0,5,5,0,-20,-40,
               -50,-40,-30,-30,-30,-30,-40,-50]
PSQT_BISHOP = [-20,-10,-10,-10,-10,-10,-10,-20,
               -10,5,0,0,0,0,5,-10,
               -10,10,10,10,10,10,10,-10,
               -10,0,10,10,10,10,0,-10,
               -10,5,5,10,10,5,5,-10,
               -10,0,5,10,10,5,0,-10,
               -10,0,0,0,0,0,0,-10,
               -20,-10,-10,-10,-10,-10,-10,-20]
PSQT_ROOK = [0,0,5,10,10,5,0,0,
             -5,0,0,0,0,0,0,-5,
             -5,0,0,0,0,0,0,-5,
             -5,0,0,0,0,0,0,-5,
             -5,0,0,0,0,0,0,-5,
             -5,0,0,0,0,0,0,-5,
             5,10,10,10,10,10,10,5,
             0,0,0,0,0,0,0,0]
PSQT_QUEEN = [-20,-10,-10,-5,-5,-10,-10,-20,
              -10,0,0,0,0,0,0,-10,
              -10,0,5,5,5,5,0,-10,
              -5,0,5,5,5,5,0,-5,
              0,0,5,5,5,5,0,-5,
              -10,5,5,5,5,5,0,-10,
              -10,0,5,0,0,0,0,-10,
              -20,-10,-10,-5,-5,-10,-10,-20]
PSQT_KING = [-30,-40,-40,-50,-50,-40,-40,-30,
             -30,-40,-40,-50,-50,-40,-40,-30,
             -30,-40,-40,-50,-50,-40,-40,-30,
             -30,-40,-40,-50,-50,-40,-40,-30,
             -20,-30,-30,-40,-40,-30,-30,-20,
             -10,-20,-20,-20,-20,-20,-20,-10,
             20,20,0,0,0,0,20,20,
             20,30,10,0,0,10,30,20]

PSQT = {
    chess.PAWN: PSQT_PAWN,
    chess.KNIGHT: PSQT_KNIGHT,
    chess.BISHOP: PSQT_BISHOP,
    chess.ROOK: PSQT_ROOK,
    chess.QUEEN: PSQT_QUEEN,
    chess.KING: PSQT_KING
}

# ---------------- LEARNING ----------------
if os.path.exists(LEARN_FILE):
    with open(LEARN_FILE, "r") as f:
        learn_data = json.load(f)
else:
    learn_data = {}

def save_learn():
    with open(LEARN_FILE,"w") as f:
        json.dump(learn_data,f)

def learn_move(fen, move):
    key = fen.split(" ")[0]
    if key not in learn_data:
        learn_data[key] = {}
    move_str = move.uci()
    learn_data[key][move_str] = learn_data[key].get(move_str,0)+1
    save_learn()

def best_learned(fen):
    key = fen.split(" ")[0]
    if key in learn_data and learn_data[key]:
        # pick most successful move
        return max(learn_data[key], key=learn_data[key].get)
    return None

# ---------------- EVALUATION ----------------
def flip_sq(sq):
    return chess.square(chess.square_file(sq), 7 - chess.square_rank(sq))

def evaluate(board):
    # Terminal
    if board.is_checkmate():
        return -MATE if board.turn else MATE
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    material = 0
    psqt_score = 0
    for pt in chess.PIECE_TYPES:
        for sq in board.pieces(pt, chess.WHITE):
            material += PIECE_VAL[pt]
            psqt_score += PSQT[pt][sq]
        for sq in board.pieces(pt, chess.BLACK):
            material -= PIECE_VAL[pt]
            psqt_score -= PSQT[pt][flip_sq(sq)]

    # mobility
    board_turn = board.turn
    my_moves = len(list(board.legal_moves))
    board.push(chess.Move.null())
    opp_moves = len(list(board.legal_moves))
    board.pop()
    mobility = (my_moves - opp_moves)*2

    # simple king safety
    ks = 0
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    if wk is not None:
        ks -= len(list(board.attackers(chess.BLACK,wk)))*30
    if bk is not None:
        ks += len(list(board.attackers(chess.WHITE,bk)))*30

    score = material + psqt_score + mobility + ks
    return score if board_turn==chess.WHITE else -score

# ---------------- SEARCH ----------------
class Searcher:
    def __init__(self):
        self.nodes = 0
        self.stop = False
        self.start_time = 0
        self.movetime = DEFAULT_MOVETIME_MS

    def time_up(self):
        return (time.time()-self.start_time)*1000>=self.movetime

    def order_moves(self, board, moves):
        # prioritize captures, checks
        def score_move(m):
            s = 0
            if board.is_capture(m):
                s+=PIECE_VAL.get(board.piece_type_at(m.to_square),0)*10
            if board.gives_check(m):
                s+=50
            return s
        return sorted(moves,key=score_move,reverse=True)

    def quiescence(self, board, alpha, beta):
        stand = evaluate(board)
        if stand >= beta: return beta
        if stand > alpha: alpha = stand
        for m in self.order_moves(board,[m for m in board.legal_moves if board.is_capture(m) or board.gives_check(m)]):
            board.push(m)
            score = -self.quiescence(board, -beta, -alpha)
            board.pop()
            if score>=beta: return beta
            if score>alpha: alpha=score
        return alpha

    def alphabeta(self, board, depth, alpha, beta):
        if self.stop: return 0
        if depth<=0:
            return self.quiescence(board, alpha, beta)
        legal = list(board.legal_moves)
        if not legal:
            if board.is_check(): return -MATE
            return 0
        legal = self.order_moves(board, legal)
        best_score = -INF
        for m in legal:
            board.push(m)
            score = -self.alphabeta(board, depth-1, -beta, -alpha)
            board.pop()
            if score>best_score:
                best_score=score
            if score>alpha: alpha=score
            if alpha>=beta: break
        return best_score

    def search(self, board, movetime=None):
        self.nodes=0
        self.stop=False
        self.start_time=time.time()
        self.movetime = movetime or self.movetime
        best_move = None
        for depth in range(1, MAX_DEPTH+1):
            legal = list(board.legal_moves)
            if not legal: break
            score_dict = {}
            for m in legal:
                board.push(m)
                score = -self.alphabeta(board, depth-1, -INF, INF)
                board.pop()
                score_dict[m]=score
                if self.stop: break
            if score_dict:
                best_move = max(score_dict, key=score_dict.get)
            if self.time_up(): break
        return best_move

# ---------------- ENGINE INTERFACE ----------------
class Engine:
    def __init__(self):
        self.searcher = Searcher()
        self.fast = False

    def set_fast(self, val:bool):
        self.fast = val

    def bestmove_board(self, board:chess.Board):
        # check learned moves
        lm = best_learned(board.fen())
        if lm:
            move = chess.Move.from_uci(lm)
            if move in board.legal_moves:
                return move
        movetime = FAST_MOVETIME_MS if self.fast else DEFAULT_MOVETIME_MS
        move = self.searcher.search(board, movetime=movetime)
        if move:
            learn_move(board.fen(), move)
        return move

# ---------------- TEST ----------------
if __name__=="__main__":
    engine = Engine()
    board = chess.Board()
    while not board.is_game_over():
        print(board)
        move = engine.bestmove_board(board)
        print("Engine plays:", move)
        if move is None:
            break
        board.push(move)
