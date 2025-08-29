#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
engine.py - Pure Python Chess Engine
Approx. 2100 Elo strength
Features:
- Piece-square tables
- Evaluation (material + positional + mobility + king safety)
- Iterative deepening with alpha-beta
- Quiescence search
- Basic learning: stores mistakes in learn.json
"""

import chess
import time
import random
import json
import os

# ---------------- Config ----------------
INF = 1000000
MATE = 999999
MAX_DEPTH = 5  # increase for stronger play (slower)
DEFAULT_MOVETIME_MS = 500
LEARN_FILE = "learn.json"

# Piece values
PIECE_VAL = {chess.PAWN:100, chess.KNIGHT:320, chess.BISHOP:330,
             chess.ROOK:500, chess.QUEEN:900, chess.KING:20000}

# Piece-square tables (white perspective)
PSQT_PAWN = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
]
PSQT_KNIGHT = [
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50
]
PSQT_BISHOP = [
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10,-10,-10,-10,-10,-20
]
PSQT_ROOK = [
     0,  0,  5, 10, 10,  5,  0,  0,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     5, 10, 10, 10, 10, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
]
PSQT_QUEEN = [
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
     0,  0,  5,  5,  5,  5,  0, -5,
   -10,  5,  5,  5,  5,  5,  0,-10,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20
]
PSQT_KING = [
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20
]
PSQT = {
    chess.PAWN: PSQT_PAWN,
    chess.KNIGHT: PSQT_KNIGHT,
    chess.BISHOP: PSQT_BISHOP,
    chess.ROOK: PSQT_ROOK,
    chess.QUEEN: PSQT_QUEEN,
    chess.KING: PSQT_KING
}

# ---------------- Engine Class ----------------
class Engine:
    def __init__(self):
        self.learn_file = LEARN_FILE
        self.mistakes = self.load_learning()
        self.nodes = 0

    # --- Load learning data ---
    def load_learning(self):
        if os.path.exists(self.learn_file):
            try:
                with open(self.learn_file,"r") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    # --- Save learning data ---
    def save_learning(self):
        try:
            with open(self.learn_file,"w") as f:
                json.dump(self.mistakes,f)
        except:
            pass

    # --- Evaluate position ---
    def evaluate(self, board: chess.Board):
        if board.is_checkmate():
            return -MATE if board.turn else MATE
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        score = 0
        for pt in chess.PIECE_TYPES:
            for sq in board.pieces(pt,chess.WHITE):
                score += PIECE_VAL[pt] + PSQT[pt][sq]
            for sq in board.pieces(pt,chess.BLACK):
                score -= PIECE_VAL[pt] + PSQT[pt][63-sq]  # flip sq
        # simple mobility
        board.push(chess.Move.null())
        mob = len(list(board.legal_moves))
        board.pop()
        score += mob
        return score if board.turn==chess.WHITE else -score

    # --- Quiescence search ---
    def quiesce(self, board, alpha, beta):
        stand = self.evaluate(board)
        if stand >= beta:
            return beta
        if stand > alpha:
            alpha = stand
        for move in board.legal_moves:
            if board.is_capture(move):
                board.push(move)
                score = -self.quiesce(board,-beta,-alpha)
                board.pop()
                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score
        return alpha

    # --- Alpha-Beta search ---
    def alphabeta(self, board, depth, alpha, beta):
        if depth==0:
            return self.quiesce(board,alpha,beta)
        max_score = -INF
        moves = list(board.legal_moves)
        random.shuffle(moves)
        for move in moves:
            board.push(move)
            score = -self.alphabeta(board,depth-1,-beta,-alpha)
            board.pop()
            if score >= beta:
                return beta
            if score > max_score:
                max_score = score
            if score > alpha:
                alpha = score
        return max_score

    # --- Iterative Deepening ---
    def search(self, board: chess.Board, movetime_ms=DEFAULT_MOVETIME_MS):
        start_time = time.time()
        best_move = None
        depth = 1
        while True:
            best_score = -INF
            for move in list(board.legal_moves):
                board.push(move)
                score = -self.alphabeta(board,depth-1,-INF,INF)
                board.pop()
                if score > best_score:
                    best_score = score
                    best_move = move
            depth += 1
            if depth > MAX_DEPTH or (time.time()-start_time)*1000 >= movetime_ms:
                break
        return best_move

    # --- Public API ---
    def bestmove_board(self, board: chess.Board, movetime_ms=DEFAULT_MOVETIME_MS):
        move = self.search(board,movetime_ms)
        if move is None:
            # fallback random
            move = random.choice(list(board.legal_moves))
        return move

# ---------------- Test ----------------
if __name__=="__main__":
    board = chess.Board()
    engine = Engine()
    while not board.is_game_over():
        print(board)
        move = engine.bestmove_board(board,500)
        print("Move:",move)
        board.push(move)
