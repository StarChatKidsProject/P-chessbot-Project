#!/usr/bin/env python3
"""
PyChess2600+ — single-file Python UCI engine using python-chess.

Features:
- Full legality via python-chess
- Iterative deepening, negamax alpha-beta, quiescence (captures+promotions, MVV-LVA)
- Transposition table with Zobrist keys (polyglot hash)
- Move ordering: TT move, MVV-LVA, checks, killers, history
- Tapered evaluation (MG/EG PST), pawn structure (doubled/isolated/passed), king safety (pawn shield),
  mobility and simple endgame scaling
- Polyglot opening book support (UCI setoption to load/enable/disable)
- Practical time management using wtime/btime/winc/binc/movetime

Usage:
  pip3 install python-chess
  python3 pychess_2600_plus.py
"""

import sys
import time
import math
from collections import namedtuple, defaultdict

import chess
import chess.polyglot

INFINITY = 10**9

# -------------------- Search params (tune as you like) --------------------
MAX_DEPTH = 8          # hard cap for iterative deepening (raise if you have time)
Q_DEPTH_CAP = 24       # quiescence node cap safeguard
TT_ENABLED = True
TT_SIZE_MB = 64        # soft target; Python dict not exact, but we purge roughly to this
USE_KILLERS = True
USE_HISTORY = True
ASPIRATION = False     # simple full windows by default (safer)
NULL_MOVE_PRUNING = False  # keep false for clarity & safety

# Move ordering weights
CHECK_BONUS = 150
PROMO_BONUS = 800

# -------------------- Piece values and PSTs (MG/EG) --------------------
VAL_P = 100
VAL_N = 320
VAL_B = 330
VAL_R = 500
VAL_Q = 900
VAL_K = 20000

PIECE_VALUE = {
    chess.PAWN: VAL_P,
    chess.KNIGHT: VAL_N,
    chess.BISHOP: VAL_B,
    chess.ROOK: VAL_R,
    chess.QUEEN: VAL_Q,
    chess.KING: VAL_K,
}

# Midgame PSTs (white perspective; mirror for black)
PST_MG = {
    chess.PAWN: [
         0,  5,  5,  0,  5, 10, 50,  0,
         0, 10, -5,  0,  5, 10, 50,  0,
         0, 10,-10,  0, 10, 20, 50,  0,
         0,-20,  0, 20, 25, 30, 50,  0,
         0,-20,  0, 20, 25, 30, 50,  0,
         0, 10,-10,  0, 10, 20, 50,  0,
         0, 10, -5,  0,  5, 10, 50,  0,
         0,  0,  0,  0,  0,  0,  0,  0,
    ],
    chess.KNIGHT: [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50,
    ],
    chess.BISHOP: [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -20,-10,-10,-10,-10,-10,-10,-20,
    ],
    chess.ROOK: [
         0,  0,  5, 10, 10,  5,  0,  0,
         0,  0,  5, 10, 10,  5,  0,  0,
         0,  0,  5, 10, 10,  5,  0,  0,
         0,  0,  5, 10, 10,  5,  0,  0,
         0,  0,  5, 10, 10,  5,  0,  0,
         0,  0,  5, 10, 10,  5,  0,  0,
         25, 25, 25, 25, 25, 25, 25, 25,
         0,  0,  5, 10, 10,  5,  0,  0,
    ],
    chess.QUEEN: [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
         -5,  0,  5,  5,  5,  5,  0, -5,
          0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20,
    ],
    chess.KING: [
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
         20, 20,  0,  0,  0,  0, 20, 20,
         20, 30, 10,  0,  0, 10, 30, 20,
    ],
}

# Endgame king PST (safer in center)
PST_EG = dict(PST_MG)
PST_EG[chess.KING] = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-40,-30,-20,-20,-30,-40,-50,
]

# -------------------- Transposition table --------------------
TTEntry = namedtuple("TTEntry", "value depth flag best")
TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2

# -------------------- Helpers --------------------
def mirror_if_black(sq, color):
    return sq if color == chess.WHITE else chess.square_mirror(sq)

def tapered_pst(piece_type, square, color, phase):
    mg = PST_MG.get(piece_type, [0]*64)[mirror_if_black(square, color)]
    eg = PST_EG.get(piece_type, [0]*64)[mirror_if_black(square, color)]
    # linear interpolation by phase (0..24 typical): mg*phase + eg*(24-phase)
    PHASE_MAX = 24
    phase = max(0, min(PHASE_MAX, phase))
    return (mg * phase + eg * (PHASE_MAX - phase)) // PHASE_MAX

def game_phase(board: chess.Board):
    # count remaining material (without pawns/kings) to estimate phase
    phase = 0
    for pt, w in [(chess.KNIGHT,1), (chess.BISHOP,1), (chess.ROOK,2), (chess.QUEEN,4)]:
        phase += w * (len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK)))
    # Normalize to ~24 max
    return min(24, int(phase * 2))

def count_pawns(board: chess.Board, color):
    return len(board.pieces(chess.PAWN, color))

# -------------------- Pawn structure / King safety --------------------
def pawn_file_mask(board: chess.Board, color):
    """Return dict: file -> count of pawns on that file for color."""
    files = {f:0 for f in range(8)}
    for sq in board.pieces(chess.PAWN, color):
        files[chess.square_file(sq)] += 1
    return files

def doubled_isolated_passed(board: chess.Board, color):
    bonus = 0
    files = pawn_file_mask(board, color)
    opp = not color
    for sq in board.pieces(chess.PAWN, color):
        file = chess.square_file(sq)
        rank = chess.square_rank(sq) if color == chess.WHITE else 7 - chess.square_rank(sq)
        # doubled
        if files[file] > 1:
            bonus -= 12
        # isolated (no friendly pawns on adjacent files)
        if (file == 0 or files[file-1] == 0) and (file == 7 or files[file+1] == 0):
            bonus -= 10
        # passed pawn: no opposite pawn ahead on same or adjacent files
        ahead_squares = []
        step = 8 if color == chess.WHITE else -8
        for r in range(rank+1, 8):
            target_rank = r if color == chess.WHITE else 7-r
            # build squares by file -1..+1
            for df in (-1, 0, 1):
                f2 = file + df
                if 0 <= f2 < 8:
                    sq2 = chess.square(f2, r if color == chess.WHITE else 7-r)
                    ahead_squares.append(sq2)
        has_opp_pawn_ahead = any(sq2 in board.pieces(chess.PAWN, opp) for sq2 in ahead_squares)
        if not has_opp_pawn_ahead:
            # encourage more as it advances
            bonus += 12 + rank * 3
    return bonus

def pawn_shield(board: chess.Board, color):
    """Crude king safety: count friendly pawns in front of king (3 files) on ranks 2-3 (for that side)."""
    ksq = board.king(color)
    if ksq is None:
        return 0
    kfile = chess.square_file(ksq)
    ranks = [1, 2] if color == chess.WHITE else [6, 5]
    files = [f for f in (kfile-1, kfile, kfile+1) if 0 <= f < 8]
    bonus = 0
    for f in files:
        for r in ranks:
            sq = chess.square(f, r)
            if board.piece_at(sq) and board.piece_at(sq).piece_type == chess.PAWN and board.piece_at(sq).color == color:
                bonus += 8
    return bonus

# -------------------- Evaluation --------------------
def evaluate(board: chess.Board):
    """Return score from side-to-move perspective (centipawns)."""
    # mate/stalemate shortcuts (optional; search usually handles)
    if board.is_checkmate():
        return -INFINITY + 1  # current side to move is mated
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    phase = game_phase(board)

    score = 0
    # material + PST tapered
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if not p:
            continue
        v = PIECE_VALUE[p.piece_type]
        pstv = tapered_pst(p.piece_type, sq, p.color, phase)
        if p.color == chess.WHITE:
            score += v + pstv
        else:
            score -= v + pstv

    # Pawn structure
    score += doubled_isolated_passed(board, chess.WHITE)
    score -= doubled_isolated_passed(board, chess.BLACK)

    # Mobility (legal move count difference)
    # Note: python-chess generates legal moves for the side to move only, so we must flip turns.
    board_turn = board.turn
    w_legal = len(list(board.legal_moves)) if board.turn == chess.WHITE else 0
    board.turn = chess.BLACK
    b_legal = len(list(board.legal_moves)) if board_turn == chess.WHITE else len(list(board.legal_moves))
    board.turn = board_turn
    score += (w_legal - b_legal) * 2

    # King safety (pawn shield)
    score += pawn_shield(board, chess.WHITE)
    score -= pawn_shield(board, chess.BLACK)

    # Encourage bishop pair
    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
        score += 15
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
        score -= 15

    # Endgame scaling: if few pieces left, shrink PST influence slightly handled by tapered already
    return score if board.turn == chess.WHITE else -score

# -------------------- Search (negamax, quiescence, TT, ordering) --------------------
class Search:
    def __init__(self):
        self.tt = {}  # hash -> TTEntry
        self.nodes = 0
        self.stop = False
        self.start_time = 0.0
        self.limit_s = 1.0
        # killers[ply] = [m1, m2]
        self.killers = defaultdict(lambda: [None, None])
        # history (move -> score) using (from,to,piece_type) tuple key
        self.history = defaultdict(int)

    def maybe_purge_tt(self):
        # crude size control to avoid unbounded dict growth
        if len(self.tt) > (TT_SIZE_MB * 1024):  # very rough heuristic
            # keep only deeper entries
            self.tt = {k:v for k,v in sorted(self.tt.items(), key=lambda kv: kv[1].depth, reverse=True)[:TT_SIZE_MB*800]}

    def time_up(self):
        return (time.time() - self.start_time) >= self.limit_s

    def mvv_lva_score(self, board: chess.Board, m: chess.Move):
        if not board.is_capture(m):
            return 0
        cap = board.piece_at(m.to_square)
        att = board.piece_at(m.from_square)
        if not cap or not att:
            return 0
        return 10 * PIECE_VALUE[cap.piece_type] - PIECE_VALUE[att.piece_type]

    def order_moves(self, board: chess.Board, moves, tt_move=None, ply=0):
        def score(m):
            s = 0
            if tt_move and m == tt_move:
                s += 10_000_000
            if board.is_capture(m):
                s += 1000 + self.mvv_lva_score(board, m)
            if m.promotion:
                s += PROMO_BONUS
            if board.gives_check(m):
                s += CHECK_BONUS
            if USE_KILLERS:
                if m == self.killers[ply][0]:
                    s += 900
                elif m == self.killers[ply][1]:
                    s += 800
            if USE_HISTORY:
                key = (m.from_square, m.to_square, board.piece_at(m.from_square).piece_type if board.piece_at(m.from_square) else 0)
                s += min(500, self.history.get(key, 0))
            return -s
        return sorted(moves, key=score)

    def quiescence(self, board: chess.Board, alpha, beta, qnodes=0):
        # Stand-pat
        stand = evaluate(board)
        if stand >= beta:
            return beta
        if stand > alpha:
            alpha = stand

        # limit
        if qnodes > Q_DEPTH_CAP:
            return alpha

        # Only captures & promotions
        captures = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
        # MVV-LVA ordering
        captures.sort(key=lambda m: -self.mvv_lva_score(board, m))

        for m in captures:
            if self.time_up():
                self.stop = True
                return alpha
            self.nodes += 1
            board.push(m)
            score = -self.quiescence(board, -beta, -alpha, qnodes+1)
            board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    def negamax(self, board: chess.Board, depth, alpha, beta, ply=0):
        alpha_orig = alpha

        # time control
        if self.time_up():
            self.stop = True
            return 0

        self.nodes += 1

        # TT probe
        key = chess.polyglot.zobrist_hash(board)
        tt_move = None
        if TT_ENABLED and key in self.tt:
            entry = self.tt[key]
            if entry.depth >= depth:
                if entry.flag == TT_EXACT:
                    return entry.value
                if entry.flag == TT_LOWER and entry.value > alpha:
                    alpha = entry.value
                elif entry.flag == TT_UPPER and entry.value < beta:
                    beta = entry.value
                if alpha >= beta:
                    return entry.value
            tt_move = entry.best

        # leaf
        if depth <= 0:
            return self.quiescence(board, alpha, beta)

        # generate & order
        moves = list(board.legal_moves)
        if not moves:
            # terminal: checkmate/stalemate
            return evaluate(board)

        moves = self.order_moves(board, moves, tt_move, ply)

        best_val = -INFINITY
        best_move = None

        for m in moves:
            if self.time_up():
                self.stop = True
                break
            board.push(m)
            val = -self.negamax(board, depth-1, -beta, -alpha, ply+1)
            board.pop()
            if self.stop:
                return 0
            if val > best_val:
                best_val = val
                best_move = m
            if best_val > alpha:
                alpha = best_val
            if alpha >= beta:
                # store killer/history for non-captures
                if USE_KILLERS and not board.is_capture(m):
                    k1, k2 = self.killers[ply]
                    if m != k1:
                        self.killers[ply] = [m, k1]
                break

        # store history for quiet best move
        if USE_HISTORY and best_move and not board.is_capture(best_move):
            key_hist = (best_move.from_square, best_move.to_square,
                        board.piece_at(best_move.from_square).piece_type if board.piece_at(best_move.from_square) else 0)
            self.history[key_hist] += depth * depth

        # store TT
        if TT_ENABLED:
            flag = TT_EXACT
            if best_val <= alpha_orig:
                flag = TT_UPPER
            elif best_val >= beta:
                flag = TT_LOWER
            self.tt[key] = TTEntry(best_val, depth, flag, best_move)
            self.maybe_purge_tt()

        return best_val

    def search(self, board: chess.Board, max_depth, time_limit_s):
        self.nodes = 0
        self.stop = False
        self.start_time = time.time()
        self.limit_s = max(0.02, time_limit_s)

        best_move = None
        best_val = -INFINITY

        # simple aspiration disabled for reliability; can be enabled
        alpha, beta = -INFINITY, INFINITY

        for depth in range(1, max_depth+1):
            if self.time_up():
                break
            val, move = self._root(board, depth, alpha, beta)
            if self.stop:
                break
            if move:
                best_move, best_val = move, val
            # optional aspiration window tuning
            if ASPIRATION and best_move is not None:
                window = 30
                alpha, beta = best_val - window, best_val + window

        return best_move, best_val

    def _root(self, board: chess.Board, depth, alpha, beta):
        best_move = None
        best_val = -INFINITY

        moves = list(board.legal_moves)
        if not moves:
            return evaluate(board), None

        # prefer TT move at root
        key = chess.polyglot.zobrist_hash(board)
        tt_move = self.tt[key].best if TT_ENABLED and key in self.tt else None
        moves = self.order_moves(board, moves, tt_move, ply=0)

        for m in moves:
            if self.time_up():
                self.stop = True
                break
            board.push(m)
            val = -self.negamax(board, depth-1, -beta, -alpha, ply=1)
            board.pop()
            if self.stop:
                break
            if val > best_val:
                best_val = val
                best_move = m
            if best_val > alpha:
                alpha = best_val

        return best_val, best_move

# -------------------- Opening book --------------------
class OpeningBook:
    def __init__(self):
        self.reader = None
        self.enabled = True
        self.path = None

    def load(self, path):
        try:
            self.reader = chess.polyglot.open_reader(path)
            self.path = path
            self.enabled = True
            return True
        except Exception:
            self.reader = None
            self.path = None
            self.enabled = False
            return False

    def close(self):
        if self.reader:
            self.reader.close()
        self.reader = None

    def best_move(self, board: chess.Board):
        if not self.enabled or not self.reader:
            return None
        try:
            entries = list(self.reader.find_all(board))
            if not entries:
                return None
            # pick weighted by polyglot weight (common practice)
            total = sum(e.weight for e in entries)
            r = (hash(board.fen()) & 0xFFFFFFFF) % max(1, total)
            acc = 0
            for e in entries:
                acc += e.weight
                if r < acc:
                    return e.move()
            return entries[0].move()
        except Exception:
            return None

# -------------------- UCI Engine --------------------
class UCIEngine:
    def __init__(self):
        self.board = chess.Board()
        self.searcher = Search()
        self.book = OpeningBook()
        self.running = True

    # time management heuristic
    def compute_time(self, parts):
        # defaults
        movetime = None
        wtime = btime = winc = binc = None
        if "movetime" in parts:
            movetime = int(parts[parts.index("movetime")+1]) / 1000.0
            return max(0.02, movetime)
        # use side-specific times
        if "wtime" in parts:
            wtime = int(parts[parts.index("wtime")+1]) / 1000.0
        if "btime" in parts:
            btime = int(parts[parts.index("btime")+1]) / 1000.0
        if "winc" in parts:
            winc = int(parts[parts.index("winc")+1]) / 1000.0
        if "binc" in parts:
            binc = int(parts[parts.index("binc")+1]) / 1000.0
        # simple allocation: 3% of remaining time + 0.7 * increment
        if self.board.turn == chess.WHITE and wtime is not None:
            inc = winc or 0.0
            return max(0.03, wtime * 0.03 + inc * 0.7)
        if self.board.turn == chess.BLACK and btime is not None:
            inc = binc or 0.0
            return max(0.03, btime * 0.03 + inc * 0.7)
        return 1.0

    def handle_position(self, parts):
        # "position [startpos | fen <FEN>] [moves ...]"
        if "startpos" in parts:
            self.board.reset()
            idx = parts.index("startpos")
            tail = parts[idx+1:]
        elif "fen" in parts:
            fen_i = parts.index("fen") + 1
            # FEN has 6 fields; capture until 'moves' or end
            end = len(parts)
            if "moves" in parts:
                end = parts.index("moves")
            fen = " ".join(parts[fen_i:end])
            try:
                self.board.set_fen(fen)
            except Exception:
                # ignore invalid fen
                self.board.reset()
            tail = parts[end+1:] if "moves" in parts else []
        else:
            tail = []

        if "moves" in parts:
            moves_idx = parts.index("moves")
            for mv in parts[moves_idx+1:]:
                try:
                    self.board.push_uci(mv)
                except Exception:
                    try:
                        self.board.push_san(mv)
                    except Exception:
                        pass

    def handle_setoption(self, parts):
        # Examples:
        # setoption name BookFile value /path/to/book.bin
        # setoption name OwnBook value true/false
        try:
            name_idx = parts.index("name") + 1
            value_idx = parts.index("value") + 1
            name = " ".join(parts[name_idx: value_idx-1]).strip().lower()
            value = " ".join(parts[value_idx:]).strip()
        except Exception:
            return

        if name == "bookfile":
            ok = self.book.load(value)
            if ok:
                print(f"info string Book loaded: {value}")
            else:
                print(f"info string Failed to load book: {value}")
        elif name in ("ownbook", "usebook"):
            v = value.strip().lower() in ("true", "1", "on", "yes")
            self.book.enabled = v
            print(f"info string OwnBook set to {self.book.enabled}")
        elif name == "hash":  # hint only; Python dict unmanaged
            print("info string Hash option acknowledged (advisory only)")
        else:
            print(f"info string Unknown option '{name}' ignored")

    def loop(self):
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cmd = parts[0]

            if cmd == "uci":
                print("id name P-chessbot")
                print("id author pGOD1000")
                print("option name BookFile type string default ")
                print("option name OwnBook type check default true")
                print("uciok")
            elif cmd == "isready":
                print("readyok")
            elif cmd == "ucinewgame":
                self.board.reset()
                self.searcher = Search()  # fresh tables
            elif cmd == "position":
                self.handle_position(parts)
            elif cmd == "setoption":
                self.handle_setoption(parts)
            elif cmd == "go":
                # try book first
                book_move = self.book.best_move(self.board)
                if book_move:
                    print("bestmove", book_move.uci())
                    sys.stdout.flush()
                    continue

                t = self.compute_time(parts)
                best, val = self.searcher.search(self.board, MAX_DEPTH, t)
                if best is None:
                    print("bestmove 0000")
                else:
                    print("bestmove", best.uci())
            elif cmd == "stop":
                # cooperative stop (not strictly needed in this simple loop)
                pass
            elif cmd == "quit":
                break

            sys.stdout.flush()

if __name__ == "__main__":
    UCIEngine().loop()
