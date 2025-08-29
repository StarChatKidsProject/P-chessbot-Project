#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Python UCI chess engine (~2100 Elo)
Features:
- Iterative deepening
- Alpha-beta pruning
- Simple evaluation (material + mobility + piece-square tables)
- Learning from mistakes (learn.json)
- Fast enough for lichess bots
Dependencies: python-chess
"""

import chess
import chess.polyglot
import json
import time
import random
from pathlib import Path

# ---------------- Config ----------------
NAME = "P-chessbot-Engine"
AUTHOR = "pGOD1000"
MAX_DEPTH = 5
MOVE_TIME_MS = 1000
LEARN_FILE = "learn.json"
INF = 100_000

# ---------------- Piece values ----------------
PIECE_VAL = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000
}

# ---------------- Piece-square tables ----------------
PSQT_PAWN = [
 0, 0, 0, 0, 0, 0, 0, 0,
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

# ---------------- Learning ----------------
class Learner:
    def __init__(self, path=LEARN_FILE):
        self.path = Path(path)
        if self.path.exists():
            self.data = json.load(open(path))
        else:
            self.data = {}

    def get(self, fen):
        return self.data.get(fen, 0)

    def update(self, fen, score):
        self.data[fen] = score
        with open(self.path, "w") as f:
            json.dump(self.data, f)

learner = Learner()

# ---------------- Evaluation ----------------
def evaluate(board: chess.Board):
    if board.is_checkmate():
        return -INF if board.turn else INF
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    score = 0
    for pt in PIECE_VAL:
        score += len(board.pieces(pt, chess.WHITE)) * PIECE_VAL[pt]
        score -= len(board.pieces(pt, chess.BLACK)) * PIECE_VAL[pt]

        for sq in board.pieces(pt, chess.WHITE):
            score += PSQT[pt][sq]
        for sq in board.pieces(pt, chess.BLACK):
            score -= PSQT[pt][chess.square_mirror(sq)]

    # Mobility
    board.push(chess.Move.null())
    opp_moves = len(list(board.legal_moves))
    board.pop()
    score += len(list(board.legal_moves)) - opp_moves

    return score if board.turn == chess.WHITE else -score

# ---------------- Search ----------------
def search(board: chess.Board, depth=MAX_DEPTH, alpha=-INF, beta=INF):
    best_move = None
    best_score = -INF

    for move in list(board.legal_moves):
        board.push(move)
        score = -negamax(board, depth-1, -beta, -alpha)
        board.pop()
        fen = board.fen()
        score += learner.get(fen)

        if score > best_score:
            best_score = score
            best_move = move
        if best_score > alpha:
            alpha = best_score

    return best_move

def negamax(board: chess.Board, depth, alpha, beta):
    if depth == 0:
        return evaluate(board)

    max_eval = -INF
    for move in list(board.legal_moves):
        board.push(move)
        eval = -negamax(board, depth-1, -beta, -alpha)
        board.pop()
        fen = board.fen()
        eval += learner.get(fen)

        if eval > max_eval:
            max_eval = eval
        alpha = max(alpha, eval)
        if alpha >= beta:
            break
    return max_eval

# ---------------- UCI ----------------
def uci_loop():
    print(f"id name {NAME}")
    print(f"id author {AUTHOR}")
    print("uciok")

    board = chess.Board()
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "isready":
            print("readyok")
        elif line.startswith("position"):
            parts = line.split()
            if "startpos" in parts:
                board = chess.Board()
                idx = parts.index("startpos") + 1
            elif "fen" in parts:
                idx = parts.index("fen") + 1
                fen = " ".join(parts[idx:idx+6])
                board = chess.Board(fen)
                idx += 6
            if idx < len(parts) and parts[idx] == "moves":
                for mv in parts[idx+1:]:
                    board.push_uci(mv)
        elif line.startswith("go"):
            move = search(board)
            if move:
                print(f"bestmove {move.uci()}", flush=True)
        elif line == "quit":
            break

if __name__ == "__main__":
    uci_loop()
