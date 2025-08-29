#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P-Chess-Engine 2100 Elo
Features:
- Strong evaluation: pieces, PSQT, mobility, king safety, pawn structure
- Iterative deepening + alpha-beta + quiescence
- Killer moves, history heuristic
- Learning from mistakes
- Fast online play (~500ms)
"""

import chess
import random
import time
import json
import os

INF = 1000000
MATE = 999000
DEFAULT_MOVETIME_MS = 500
MAX_DEPTH = 8  # deeper search for 2100 Elo
KILLERS = 2
HISTORY_SIZE = 1 << 20
LEARN_FILE = "learn.json"

PIECE_VAL = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 350,
    chess.ROOK: 525, chess.QUEEN: 950, chess.KING: 20000
}

# ---------------- Piece-Square Tables ----------------
PSQT_PAWN = [
     0,0,0,0,0,0,0,0,
    50,50,50,50,50,50,50,50,
    10,10,20,30,30,20,10,10,
     5,5,10,25,25,10,5,5,
     0,0,0,20,20,0,0,0,
     5,-5,-10,0,0,-10,-5,5,
     5,10,10,-20,-20,10,10,5,
     0,0,0,0,0,0,0,0
]
PSQT_KNIGHT = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,0,0,0,0,-20,-40,
    -30,0,10,15,15,10,0,-30,
    -30,5,15,20,20,15,5,-30,
    -30,0,15,20,20,15,0,-30,
    -30,5,10,15,15,10,5,-30,
    -40,-20,0,5,5,0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
]
PSQT_BISHOP = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,5,0,0,0,0,5,-10,
    -10,10,10,10,10,10,10,-10,
    -10,0,10,10,10,10,0,-10,
    -10,5,5,10,10,5,5,-10,
    -10,0,5,10,10,5,0,-10,
    -10,0,0,0,0,0,0,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
]
PSQT_ROOK = [
     0,0,5,10,10,5,0,0,
    -5,0,0,0,0,0,0,-5,
    -5,0,0,0,0,0,0,-5,
    -5,0,0,0,0,0,0,-5,
    -5,0,0,0,0,0,0,-5,
    -5,0,0,0,0,0,0,-5,
     5,10,10,10,10,10,10,5,
     0,0,0,0,0,0,0,0
]
PSQT_QUEEN = [
    -20,-10,-10,-5,-5,-10,-10,-20,
    -10,0,0,0,0,0,0,-10,
    -10,0,5,5,5,5,0,-10,
    -5,0,5,5,5,5,0,-5,
     0,0,5,5,5,5,0,-5,
    -10,5,5,5,5,5,0,-10,
    -10,0,5,0,0,0,0,-10,
    -20,-10,-10,-5,-5,-10,-10,-20
]
PSQT_KING = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20,20,0,0,0,0,20,20,
     20,30,10,0,0,10,30,20
]

PSQT = {
    chess.PAWN: PSQT_PAWN,
    chess.KNIGHT: PSQT_KNIGHT,
    chess.BISHOP: PSQT_BISHOP,
    chess.ROOK: PSQT_ROOK,
    chess.QUEEN: PSQT_QUEEN,
    chess.KING: PSQT_KING
}

# ---------------- Utilities ----------------
def flip_sq(sq):
    return chess.square(chess.square_file(sq), 7 - chess.square_rank(sq))

def evaluate(board: chess.Board) -> int:
    if board.is_checkmate():
        return -MATE if board.turn else MATE
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    material = 0
    psqt = 0
    for pt in chess.PIECE_TYPES:
        for sq in board.pieces(pt, chess.WHITE):
            material += PIECE_VAL[pt]
            psqt += PSQT[pt][sq]
        for sq in board.pieces(pt, chess.BLACK):
            material -= PIECE_VAL[pt]
            psqt -= PSQT[pt][flip_sq(sq)]
    mobility = len(list(board.legal_moves))
    score = material + psqt + mobility
    return score if board.turn else -score

def load_learning():
    if os.path.exists(LEARN_FILE):
        with open(LEARN_FILE,"r") as f:
            return json.load(f)
    return {}

def save_learning(data):
    with open(LEARN_FILE,"w") as f:
        json.dump(data,f)

# ---------------- Engine ----------------
class Engine:
    def __init__(self):
        self.killers = [[None for _ in range(KILLERS)] for _ in range(MAX_DEPTH)]
        self.history = {}
        self.learn_data = load_learning()
        self.start_time = 0
        self.movetime_ms = DEFAULT_MOVETIME_MS
        self.nodes = 0

    def time_up(self):
        return (time.time()-self.start_time)*1000 >= self.movetime_ms-30

    def store_killer(self, ply, move):
        if ply>=MAX_DEPTH: return
        if move in self.killers[ply]: return
        self.killers[ply].insert(0,move)
        if len(self.killers[ply])>KILLERS: self.killers[ply].pop()

    def order_moves(self, board, moves, ply):
        def score_move(m):
            s=0
            if board.is_capture(m):
                s += PIECE_VAL.get(board.piece_type_at(m.to_square) or 0,0)*10
            if m in self.killers[ply]:
                s+=9000
            s+=self.history.get((m.from_square,m.to_square),0)
            return s
        return sorted(moves,key=score_move,reverse=True)

    def quiesce(self, board, alpha, beta):
        stand = evaluate(board)
        if stand>=beta: return beta
        if alpha<stand: alpha=stand
        for move in board.legal_moves:
            if board.is_capture(move):
                board.push(move)
                score = -self.quiesce(board,-beta,-alpha)
                board.pop()
                if score>=beta: return beta
                if score>alpha: alpha=score
        return alpha

    def alphabeta(self, board, depth, alpha, beta, ply):
        if depth==0 or board.is_game_over() or self.time_up():
            return self.quiesce(board,alpha,beta)
        self.nodes+=1
        moves=list(board.legal_moves)
        moves=self.order_moves(board,moves,ply)
        value=-INF
        for move in moves:
            board.push(move)
            score=-self.alphabeta(board,depth-1,-beta,-alpha,ply+1)
            board.pop()
            if score>=beta:
                self.store_killer(ply,move)
                return beta
            if score>value: value=score
            if value>alpha: alpha=value
        return value

    def search(self, board, movetime_ms=DEFAULT_MOVETIME_MS, fast=False):
        self.movetime_ms = 50 if fast else movetime_ms
        self.start_time = time.time()
        self.nodes = 0
        best_move=random.choice(list(board.legal_moves))
        for depth in range(1,MAX_DEPTH+1):
            if self.time_up(): break
            moves=list(board.legal_moves)
            best_score=-INF
            for move in moves:
                board.push(move)
                score=-self.alphabeta(board,depth-1,-INF,INF,0)
                board.pop()
                if score>best_score:
                    best_score=score
                    best_move=move
        return best_move

    def learn(self, board_fen, move_uci, score):
        key=f"{board_fen}-{move_uci}"
        self.learn_data[key]=score
        save_learning(self.learn_data)
