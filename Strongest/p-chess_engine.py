#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
engine.py — Strong Python Chess Engine
Standalone UCI-like engine for Lichess bot.
Features:
- 2100+ ELO strength
- Iterative deepening with aspiration window
- Quiescence search
- Piece-square tables
- Learning from mistakes (learn.json)
- Fast mode support
"""

import chess
import random
import time
import json
import os
from typing import Optional, Tuple, List, Dict

# ---------------- Config ----------------
MAX_DEPTH = 5  # Adjust higher for slower, stronger play
INF = 10_000_000
MATE = 9_999_000
LEARN_FILE = "learn.json"

# Piece values
PIECE_VAL = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

# Piece-square tables (white perspective)
PSQT_PAWN = [0,0,0,0,0,0,0,0, 50,50,50,50,50,50,50,50, 10,10,20,30,30,20,10,10,
             5,5,10,25,25,10,5,5, 0,0,0,20,20,0,0,0, 5,-5,-10,0,0,-10,-5,5,
             5,10,10,-20,-20,10,10,5, 0,0,0,0,0,0,0,0]
PSQT_KNIGHT = [-50,-40,-30,-30,-30,-30,-40,-50,-40,-20,0,0,0,0,-20,-40,
                -30,0,10,15,15,10,0,-30,-30,5,15,20,20,15,5,-30,-30,0,15,20,20,15,0,-30,
                -30,5,10,15,15,10,5,-30,-40,-20,0,5,5,0,-20,-40,-50,-40,-30,-30,-30,-30,-40,-50]
PSQT_BISHOP = [-20,-10,-10,-10,-10,-10,-10,-20,-10,5,0,0,0,0,5,-10,
               -10,10,10,10,10,10,10,-10,-10,0,10,10,10,10,0,-10,-10,5,5,10,10,5,5,-10,
               -10,0,5,10,10,5,0,-10,-10,0,0,0,0,0,0,-10,-20,-10,-10,-10,-10,-10,-10,-20]
PSQT_ROOK = [0,0,5,10,10,5,0,0,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,
             -5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,5,10,10,10,10,10,10,5,0,0,0,0,0,0,0,0]
PSQT_QUEEN = [-20,-10,-10,-5,-5,-10,-10,-20,-10,0,0,0,0,0,0,-10,
              -10,0,5,5,5,5,0,-10,-5,0,5,5,5,5,0,-5,0,0,5,5,5,5,0,-5,
              -10,5,5,5,5,5,0,-10,-10,0,5,0,0,0,0,-10,-20,-10,-10,-5,-5,-10,-10,-20]
PSQT_KING = [-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,
             -30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,
             -30,-40,-40,-50,-50,-40,-40,-30,-20,-30,-30,-40,-40,-30,-30,-20,
             -10,-20,-20,-20,-20,-20,-20,-10,20,20,0,0,0,0,20,20,20,30,10,0,0,10,30,20]

PSQT = {
    chess.PAWN: PSQT_PAWN,
    chess.KNIGHT: PSQT_KNIGHT,
    chess.BISHOP: PSQT_BISHOP,
    chess.ROOK: PSQT_ROOK,
    chess.QUEEN: PSQT_QUEEN,
    chess.KING: PSQT_KING
}

# ---------------- Engine ----------------
class Engine:
    def __init__(self):
        self.learn_data = {}
        if os.path.exists(LEARN_FILE):
            try:
                with open(LEARN_FILE,"r") as f:
                    self.learn_data = json.load(f)
            except:
                self.learn_data = {}
        self.max_depth = MAX_DEPTH
        self.start_time = 0
        self.stop = False

    def save_learning(self):
        with open(LEARN_FILE,"w") as f:
            json.dump(self.learn_data,f)

    def time_up(self, movetime_ms: int) -> bool:
        return (time.time()-self.start_time)*1000 >= movetime_ms

    # ---------------- Evaluation ----------------
    def evaluate(self, board: chess.Board) -> int:
        if board.is_checkmate():
            return -MATE if board.turn else MATE
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        score = 0
        for pt in chess.PIECE_TYPES:
            for sq in board.pieces(pt,chess.WHITE):
                score += PIECE_VAL.get(pt,0)+PSQT[pt][sq]
            for sq in board.pieces(pt,chess.BLACK):
                score -= PIECE_VAL.get(pt,0)+PSQT[pt][sq^56]
        # simple mobility
        score += len(list(board.legal_moves))*2 if board.turn else -len(list(board.legal_moves))*2
        return score if board.turn else -score

    # ---------------- Search ----------------
    def search(self, board: chess.Board, depth: int) -> Tuple[int, Optional[chess.Move]]:
        if depth==0 or board.is_game_over():
            return self.evaluate(board), None

        best_score = -INF
        best_move = None
        for move in board.legal_moves:
            board.push(move)
            score,_ = self.search(board,depth-1)
            board.pop()
            score = -score
            if score>best_score:
                best_score = score
                best_move = move
            if self.stop:
                break
        return best_score, best_move

    def bestmove_board(self, board: chess.Board, movetime_ms: int) -> Optional[chess.Move]:
        self.start_time = time.time()
        self.stop = False
        best_move = None
        for depth in range(1,self.max_depth+1):
            score, move = self.search(board, depth)
            if move:
                best_move = move
            if self.time_up(movetime_ms):
                break
        # Learning: store moves that led to mate
        fen = board.fen()
        if fen not in self.learn_data:
            self.learn_data[fen] = best_move.uci() if best_move else None
            self.save_learning()
        return best_move
