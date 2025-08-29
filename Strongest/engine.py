#!/usr/bin/env python3
# py2000_engine.py — compact Python UCI engine with small opening book
# Requires: python-chess

import sys
import time
import math
import threading
import random
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import chess

# make stdout line-buffered
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# ---------------- Config ----------------
ENGINE_NAME = "py2000"
ENGINE_AUTHOR = "pGOD1000"

MAX_MOVETIME_MS = 5000
MIN_MOVETIME_MS = 20
BASE_DEPTH = 6
MAX_DEPTH_CAP = 22
ASP_WINDOW = 35
QS_DEPTH_CAP = 14
DELTA_PRUNE_MARGIN = 150
TT_SIZE_MB = 32
TT_ENTRIES = max(1024, (TT_SIZE_MB * 1024 * 1024) // 40)
KILLERS_PER_PLY = 2
HISTORY_MAX = 1 << 30

INF = 10_000_000
MATE_SCORE = 9_999_000

# ---------- small opening book (maps SAN/uci lines from startpos) ----------
# This is a tiny built-in book: mapping FEN->list of uci moves
BOOK = {}
# startpos typical moves
BOOK["startpos"] = [
    "e2e4","d2d4","c2c4","g1f3","e2e3","d2d3","g1h3"  # priorities: main first
]
# some common replies (after 1.e4)
BOOK["rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"] = ["e7e5","c7c5","e7e6","c7c6","g8f6"]
# small lines added for variety (you can expand)
# NOTE: book uses raw fen strings (lack of move history); we check by FEN and "startpos"

# ---------------- helper ----------------
def _safe_int(tok: Optional[str], default: int = 0) -> int:
    if tok is None:
        return default
    s = str(tok).strip().lower()
    if not s:
        return default
    try:
        return int(s)
    except Exception:
        pass
    if s.endswith("ms"):
        try:
            return int(s[:-2])
        except Exception:
            return default
    if s.endswith("s") or s.endswith("sec") or s.endswith("secs"):
        digits = "".join(ch for ch in s if ch.isdigit())
        try:
            return int(digits) * 1000 if digits else default
        except Exception:
            return default
    digits = "".join(ch for ch in s if ch.isdigit())
    try:
        return int(digits) if digits else default
    except Exception:
        return default

def is_mate_score(score: int) -> bool:
    return abs(score) > MATE_SCORE - 1000

def cp_to_mate_text(score: int) -> Optional[str]:
    if score > MATE_SCORE - 1000:
        ply = MATE_SCORE - score
        return f"# {math.ceil(ply/2)}"
    if score < -MATE_SCORE + 1000:
        ply = MATE_SCORE + score
        return f"# -{math.ceil(ply/2)}"
    return None

# ---------------- evaluation tables ----------------
PV = {
    chess.PAWN:   (82, 94),
    chess.KNIGHT: (320, 320),
    chess.BISHOP: (330, 330),
    chess.ROOK:   (500, 500),
    chess.QUEEN:  (900, 900),
    chess.KING:   (0, 0),
}

# (Truncated but reasonable PSQT values)
# For brevity keep compact but effective tables (white perspective)
PSQT_PAWN_MG = [
 0,0,0,0,0,0,0,0,
 98,134,61,95,68,126,34,-11,
 -6,7,26,31,65,56,25,-20,
 -14,13,6,21,23,12,17,-23,
 -27,-2,-5,12,17,6,10,-25,
 -26,-4,-4,-10,3,3,33,-12,
 -35,-1,-20,-23,-15,24,38,-22,
 0,0,0,0,0,0,0,0
]
PSQT_PAWN_EG = PSQT_PAWN_MG[:]  # reuse for simplicity

PSQT_KNIGHT_MG = [
 -167,-89,-34,-49,61,-97,-15,-107,
 -73,-41,72,36,23,62,7,-17,
 -47,60,37,65,84,129,73,44,
 -9,17,19,53,37,69,18,22,
 -13,4,16,13,28,19,21,-8,
 -23,-9,12,10,19,17,25,-16,
 -29,-53,-12,-3,-1,18,-14,-19,
 -105,-21,-58,-33,-17,-28,-19,-23
]
PSQT_KNIGHT_EG = PSQT_KNIGHT_MG[:]

PSQT_BISHOP_MG = PSQT_BISHOP_EG = PSQT_KNIGHT_MG[:]
PSQT_ROOK_MG = PSQT_ROOK_EG = PSQT_KNIGHT_MG[:]
PSQT_QUEEN_MG = PSQT_QUEEN_EG = PSQT_KNIGHT_MG[:]
PSQT_KING_MG = PSQT_KING_EG = PSQT_KNIGHT_MG[:]

PSQT = {
    chess.PAWN:   (PSQT_PAWN_MG,   PSQT_PAWN_EG),
    chess.KNIGHT: (PSQT_KNIGHT_MG, PSQT_KNIGHT_EG),
    chess.BISHOP: (PSQT_BISHOP_MG, PSQT_BISHOP_EG),
    chess.ROOK:   (PSQT_ROOK_MG,   PSQT_ROOK_EG),
    chess.QUEEN:  (PSQT_QUEEN_MG,  PSQT_QUEEN_EG),
    chess.KING:   (PSQT_KING_MG,   PSQT_KING_EG),
}

PHASE_WEIGHTS = {chess.PAWN:0, chess.KNIGHT:1, chess.BISHOP:1, chess.ROOK:2, chess.QUEEN:4}
MAX_PHASE = 16

def flip_sq(sq: int) -> int:
    return chess.square(chess.square_file(sq), 7 - chess.square_rank(sq))

# ---------------- evaluation ----------------
def evaluate(board: chess.Board) -> int:
    if board.is_checkmate():
        return -MATE_SCORE + board.fullmove_number
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    mg = 0
    eg = 0
    phase = 0

    for pt in chess.PIECE_TYPES:
        pv_mg, pv_eg = PV.get(pt, (0,0))
        ps_mg, ps_eg = PSQT.get(pt, ([0]*64,[0]*64))
        for sq in board.pieces(pt, chess.WHITE):
            mg += pv_mg + ps_mg[sq]
            eg += pv_eg + ps_eg[sq]
            phase += PHASE_WEIGHTS.get(pt, 0)
        for sq in board.pieces(pt, chess.BLACK):
            sqf = flip_sq(sq)
            mg -= pv_mg + ps_mg[sqf]
            eg -= pv_eg + ps_eg[sqf]
            phase += PHASE_WEIGHTS.get(pt, 0)

    phase = min(phase, MAX_PHASE)
    score = (mg * phase + eg * (MAX_PHASE - phase)) // MAX_PHASE

    # mobility small bonus
    tmp = board.copy(stack=False)
    tmp.turn = chess.WHITE
    mob_w = sum(1 for _ in tmp.legal_moves)
    tmp.turn = chess.BLACK
    mob_b = sum(1 for _ in tmp.legal_moves)
    score += 2 * (mob_w - mob_b)

    score += pawn_structure_eval(board)
    score += king_safety_eval(board)

    return score if board.turn == chess.WHITE else -score

def pawn_structure_eval(board: chess.Board) -> int:
    score = 0
    filesW = [0]*8
    filesB = [0]*8
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        filesW[chess.square_file(sq)] += 1
    for sq in board.pieces(chess.PAWN, chess.BLACK):
        filesB[chess.square_file(sq)] += 1
    # isolated pawns penalty
    for f in range(8):
        if filesW[f] > 0 and ( (f==0 or filesW[f-1]==0) and (f==7 or filesW[f+1]==0) ):
            score -= 12 * filesW[f]
        if filesB[f] > 0 and ( (f==0 or filesB[f-1]==0) and (f==7 or filesB[f+1]==0) ):
            score += 12 * filesB[f]
    # passed pawns bonus
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        if is_passed_pawn(board, sq, chess.WHITE):
            score += 10 + 6 * chess.square_rank(sq)
    for sq in board.pieces(chess.PAWN, chess.BLACK):
        if is_passed_pawn(board, sq, chess.BLACK):
            score -= 10 + 6 * (7 - chess.square_rank(sq))
    return score

def is_passed_pawn(board: chess.Board, sq: int, color: bool) -> bool:
    file = chess.square_file(sq)
    rank = chess.square_rank(sq)
    if color == chess.WHITE:
        for f in range(max(0,file-1), min(7,file+1)+1):
            for r in range(rank+1,8):
                p = board.piece_at(chess.square(f,r))
                if p and p.piece_type == chess.PAWN and p.color == chess.BLACK:
                    return False
        return True
    else:
        for f in range(max(0,file-1), min(7,file+1)+1):
            for r in range(0,rank):
                p = board.piece_at(chess.square(f,r))
                if p and p.piece_type == chess.PAWN and p.color == chess.WHITE:
                    return False
        return True

def king_safety_eval(board: chess.Board) -> int:
    score = 0
    def file_has_pawn(f:int, color:bool)->bool:
        for r in range(8):
            p = board.piece_at(chess.square(f,r))
            if p and p.piece_type==chess.PAWN and p.color==color:
                return True
        return False
    kW = board.king(chess.WHITE)
    kB = board.king(chess.BLACK)
    if kW is not None:
        f = chess.square_file(kW)
        open_file = not file_has_pawn(f, chess.WHITE) and not file_has_pawn(f, chess.BLACK)
        semi_open = not file_has_pawn(f, chess.WHITE) and file_has_pawn(f, chess.BLACK)
        if open_file: score -= 12
        if semi_open: score -= 6
    if kB is not None:
        f = chess.square_file(kB)
        open_file = not file_has_pawn(f, chess.WHITE) and not file_has_pawn(f, chess.BLACK)
        semi_open = not file_has_pawn(f, chess.BLACK) and file_has_pawn(f, chess.WHITE)
        if open_file: score += 12
        if semi_open: score += 6
    return score

# ---------------- Transposition Table ----------------
@dataclass
class TTEntry:
    key: int
    depth: int
    score: int
    flag: int  # -1 upper, 0 exact, +1 lower
    best_move: Optional[chess.Move]

class TranspositionTable:
    def __init__(self):
        self.size = TT_ENTRIES
        self.table: List[Optional[TTEntry]] = [None] * self.size
    def put(self, key:int, entry:TTEntry):
        idx = key % self.size
        cur = self.table[idx]
        if cur is None or entry.depth >= cur.depth:
            self.table[idx] = entry
    def get(self, key:int) -> Optional[TTEntry]:
        return self.table[key % self.size]

# ---------------- Search ----------------
class Search:
    def __init__(self):
        self.tt = TranspositionTable()
        self.killers: List[List[Optional[chess.Move]]] = [[None]*KILLERS_PER_PLY for _ in range(MAX_DEPTH_CAP+QS_DEPTH_CAP+8)]
        self.history: Dict[Tuple[int,int], int] = {}
        self.stop_flag = False
        self.start_time = 0.0
        self.movetime_ms = MAX_MOVETIME_MS
        self.best_move: Optional[chess.Move] = None
        self.nodes = 0

    def should_stop(self) -> bool:
        return self.stop_flag or (time.time() - self.start_time) * 1000.0 >= self.movetime_ms

    def order_moves(self, board:chess.Board, moves:List[chess.Move], tt_move:Optional[chess.Move], ply:int)->List[chess.Move]:
        def score_mv(m):
            s = 0
            if tt_move and m == tt_move: s += 10_000_000
            if board.is_capture(m):
                victim = board.piece_type_at(m.to_square) or 0
                attacker = board.piece_type_at(m.from_square) or 0
                s += 100_000 + 10 * victim - attacker
            # killers
            for km in self.killers[ply]:
                if km and m == km: s += 50_000
            s += self.history.get((m.from_square, m.to_square), 0)
            if m.promotion:
                s += 80_000 + 10 * (PV.get(m.promotion,(0,0))[0])
            return s
        return sorted(moves, key=score_mv, reverse=True)

    def quiescence(self, board:chess.Board, alpha:int, beta:int, ply:int)->int:
        self.nodes += 1
        if self.should_stop(): return evaluate(board)
        stand = evaluate(board)
        if stand >= beta: return beta
        if alpha < stand: alpha = stand

        moves = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
        def qk(m):
            s = 0
            if m.promotion: s += 90000
            if board.is_capture(m):
                v = board.piece_type_at(m.to_square) or 0
                a = board.piece_type_at(m.from_square) or 0
                s += 1000 * (v+1) - a
            return s
        moves.sort(key=qk, reverse=True)

        def piece_val(pt:Optional[int])->int:
            if not pt: return 0
            return PV.get(pt,(0,0))[0]

        for m in moves:
            cap_piece = board.piece_type_at(m.to_square)
            if not m.promotion and cap_piece:
                if stand + piece_val(cap_piece) + DELTA_PRUNE_MARGIN <= alpha:
                    continue
            board.push(m)
            score = -self.quiescence(board, -beta, -alpha, ply+1)
            board.pop()
            if self.should_stop(): return evaluate(board)
            if score >= beta: return beta
            if score > alpha: alpha = score
        return alpha

    def alphabeta(self, board:chess.Board, depth:int, alpha:int, beta:int, ply:int)->int:
        self.nodes += 1
        if self.should_stop(): return evaluate(board)
        key = board.transposition_key()
        tt = self.tt.get(key)
        if tt and tt.depth >= depth:
            if tt.flag == 0: return tt.score
            if tt.flag == -1 and tt.score <= alpha: return tt.score
            if tt.flag == 1 and tt.score >= beta: return tt.score

        if depth == 0:
            return self.quiescence(board, alpha, beta, ply)

        if board.is_repetition(2) or board.is_stalemate():
            return 0

        legal = list(board.legal_moves)
        if not legal:
            if board.is_check(): return -MATE_SCORE + ply
            return 0

        tt_move = tt.best_move if tt else None
        best_move = None
        value = -INF
        moved = 0
        orig_alpha = alpha

        moves = self.order_moves(board, legal, tt_move, ply)
        for mv in moves:
            board.push(mv)
            nd = depth - 1
            if moved > 4 and not board.is_check() and not board.is_capture(mv):
                nd = max(0, nd - 1)
            score = -self.alphabeta(board, nd, -beta, -alpha, ply+1)
            board.pop()
            if score > value:
                value = score
                best_move = mv
                if score > alpha:
                    alpha = score
                    if not board.is_capture(mv):
                        ks = self.killers[ply]
                        if mv not in ks:
                            ks.pop()
                            ks.insert(0, mv)
                        key_hist = (mv.from_square, mv.to_square)
                        self.history[key_hist] = min(HISTORY_MAX, self.history.get(key_hist, 0) + depth * depth)
                    # cutoff
                    if alpha >= beta:
                        break
            moved += 1

        if best_move is None:
            node_score = -MATE_SCORE + ply if board.is_check() else 0
            self.tt.put(key, TTEntry(key, depth, node_score, 0, None))
            return node_score

        if value <= orig_alpha: flag = -1
        elif value >= beta: flag = 1
        else: flag = 0
        self.tt.put(key, TTEntry(key, depth, value, flag, best_move))

        if ply == 0:
            self.best_move = best_move
        return value

    def pv_line(self, board:chess.Board, depth:int)->str:
        line=[]
        b=board.copy(stack=False)
        for _ in range(depth):
            entry=self.tt.get(b.transposition_key())
            if not entry or not entry.best_move or not b.is_legal(entry.best_move):
                break
            line.append(entry.best_move.uci())
            b.push(entry.best_move)
        return " ".join(line)

    def search(self, board:chess.Board, max_depth:int)->Tuple[Optional[chess.Move], int]:
        self.stop_flag=False
        self.start_time=time.time()
        self.nodes=0
        alpha=-INF; beta=INF
        best=None; last_score=0

        for depth in range(1, max_depth+1):
            if self.should_stop(): break
            # aspiration window around last_score
            if depth > 2 and not is_mate_score(last_score):
                window = ASP_WINDOW + 10*(depth//4)
                alpha = max(-INF, last_score - window)
                beta  = min(INF, last_score + window)
            else:
                alpha, beta = -INF, INF

            score = self.alphabeta(board, depth, alpha, beta, 0)
            # if out of window, re-search full
            if score <= alpha or score >= beta:
                score = self.alphabeta(board, depth, -INF, INF, 0)
            if self.should_stop(): break
            last_score = score
            best = self.best_move

            elapsed = time.time() - self.start_time
            nps = int(self.nodes / max(1e-6, elapsed))
            pv_line = self.pv_line(board, depth)
            sc_mate = cp_to_mate_text(score)
            if sc_mate:
                print(f"info depth {depth} nodes {self.nodes} nps {nps} score mate {sc_mate.split()[-1]} pv {pv_line}", flush=True)
            else:
                print(f"info depth {depth} nodes {self.nodes} nps {nps} score cp {score} pv {pv_line}", flush=True)

            if self.should_stop(): break

        return best, last_score

# ---------------- UCI Engine ----------------
class UCIEngine:
    def __init__(self):
        self.board = chess.Board()
        self.search = Search()
        self.search_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

    def uci(self):
        print(f"id name {ENGINE_NAME}", flush=True)
        print(f"id author {ENGINE_AUTHOR}", flush=True)
        print("uciok", flush=True)

    def isready(self):
        print("readyok", flush=True)

    def ucinewgame(self):
        self.board.reset()
        self.search = Search()

    def position(self, tokens:List[str]):
        it = iter(tokens)
        first = next(it, None)
        if first == "startpos":
            self.board = chess.Board()
        elif first == "fen":
            fen_parts = []
            for _ in range(6):
                p = next(it, None)
                if p is None: break
                fen_parts.append(p)
            fen = " ".join(fen_parts)
            try:
                self.board = chess.Board(fen)
            except Exception:
                self.board = chess.Board()
        # consume until moves or end
        for tok in it:
            if tok == "moves":
                break
        for mv in it:
            try:
                move = chess.Move.from_uci(mv)
                if move in self.board.legal_moves:
                    self.board.push(move)
            except Exception:
                pass

    def stop(self):
        try:
            self.search.stop_flag = True
            if self.search_thread and self.search_thread.is_alive():
                self.search_thread.join(timeout=0.1)
        except Exception:
            pass

    def _book_move(self):
        # Try book: exact fen or startpos
        fen = self.board.fen()
        if fen in BOOK and BOOK[fen]:
            # pick a random book move among candidates that are legal
            cands = [m for m in BOOK[fen] if chess.Move.from_uci(m) in self.board.legal_moves]
            if cands:
                return chess.Move.from_uci(random.choice(cands))
        # try startpos book
        if self.board.move_stack == [] and "startpos" in BOOK:
            cands = [m for m in BOOK["startpos"] if chess.Move.from_uci(m) in self.board.legal_moves]
            if cands:
                return chess.Move.from_uci(random.choice(cands))
        return None

    def _run_search(self, max_depth:int, movetime_ms:int):
        with self.lock:
            try:
                # opening book check
                bm = self._book_move()
                if bm is not None:
                    print(f"bestmove {bm.uci()}", flush=True)
                    return

                self.search.movetime_ms = movetime_ms
                start_clock = time.time()
                # iterative deepening but also respect movetime_ms
                best = None
                depth_done = 0
                for depth in range(1, max_depth+1):
                    if (time.time() - start_clock) * 1000.0 >= movetime_ms:
                        break
                    # give search a small slice: update movetime_ms remaining
                    self.search.movetime_ms = max(1, movetime_ms - int((time.time() - start_clock) * 1000.0))
                    bm_move, score = self.search.search(self.board.copy(stack=False), depth)
                    if bm_move is not None:
                        best = bm_move
                        depth_done = depth
                    # if time exhausted, break
                    if self.search.should_stop():
                        break
                # final fallback: TT entry
                if best is None:
                    tt_entry = self.search.tt.get(self.board.transposition_key())
                    if tt_entry and tt_entry.best_move and tt_entry.best_move in self.board.legal_moves:
                        best = tt_entry.best_move
                # final fallback: first legal
                if best is None:
                    legal = list(self.board.legal_moves)
                    if legal:
                        best = legal[0]
                if best is not None:
                    print(f"bestmove {best.uci()}", flush=True)
                else:
                    print("bestmove 0000", flush=True)
            except Exception as e:
                print(f"info string exception in search {e}", flush=True)
                try:
                    legal = list(self.board.legal_moves)
                    if legal:
                        print(f"bestmove {legal[0].uci()}", flush=True)
                    else:
                        print("bestmove 0000", flush=True)
                except Exception:
                    print("bestmove 0000", flush=True)

    def go(self, tokens:List[str]):
        movetime = None; wtime = None; btime = None; winc = 0; binc = 0; depth = BASE_DEPTH; movestogo = None
        it = iter(tokens)
        for tok in it:
            if tok == "movetime":
                movetime = _safe_int(next(it, None), 0)
            elif tok == "wtime":
                wtime = _safe_int(next(it, None), 0)
            elif tok == "btime":
                btime = _safe_int(next(it, None), 0)
            elif tok == "winc":
                winc = _safe_int(next(it, None), 0)
            elif tok == "binc":
                binc = _safe_int(next(it, None), 0)
            elif tok == "depth":
                depth = max(1, min(MAX_DEPTH_CAP, _safe_int(next(it, None), BASE_DEPTH)))
            elif tok == "movestogo":
                movestogo = max(1, _safe_int(next(it, None), 0))
            elif tok == "infinite":
                movetime = None
            else:
                pass

        if movetime is not None and movetime > 0:
            movetime_ms = max(MIN_MOVETIME_MS, min(MAX_MOVETIME_MS, movetime))
        else:
            side = self.board.turn
            if side == chess.WHITE and wtime is not None:
                budget = wtime; inc = winc
            elif side == chess.BLACK and btime is not None:
                budget = btime; inc = binc
            else:
                budget = MAX_MOVETIME_MS; inc = 0
            if movestogo and movestogo > 0:
                movetime_ms = max(MIN_MOVETIME_MS, min(MAX_MOVETIME_MS, (budget // movestogo) + inc))
            else:
                try:
                    movetime_ms = max(MIN_MOVETIME_MS, min(MAX_MOVETIME_MS, int(budget * 0.03) + inc))
                except Exception:
                    movetime_ms = MAX_MOVETIME_MS

        movetime_ms = max(MIN_MOVETIME_MS, min(MAX_MOVETIME_MS, movetime_ms))
        max_depth = min(MAX_DEPTH_CAP, depth if depth > 0 else BASE_DEPTH)

        try:
            self.stop()
            # start search thread
            self.search_thread = threading.Thread(target=self._run_search, args=(max_depth, movetime_ms), daemon=True)
            self.search_thread.start()
        except Exception as e:
            print(f"info string exception starting search {e}", flush=True)
            try:
                legal = list(self.board.legal_moves)
                if legal:
                    print(f"bestmove {legal[0].uci()}", flush=True)
                else:
                    print("bestmove 0000", flush=True)
            except Exception:
                print("bestmove 0000", flush=True)

# ---------------- main ----------------
def main():
    engine = UCIEngine()
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cmd = parts[0]
            if cmd == "uci":
                engine.uci()
            elif cmd == "isready":
                engine.isready()
            elif cmd == "ucinewgame":
                engine.ucinewgame()
            elif cmd == "position":
                engine.position(parts[1:])
            elif cmd == "go":
                engine.go(parts[1:])
            elif cmd == "stop":
                engine.stop()
            elif cmd == "quit":
                engine.stop()
                break
            elif cmd == "setoption":
                # ignore for now
                pass
            else:
                pass
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"info string exception {e}", flush=True)

if __name__ == "__main__":
    main()
