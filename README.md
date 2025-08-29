📖 P-chessbot-Engine – A Python Chess Engine

P-chessbot-Engine is a single-file chess engine written in pure Python using the python-chess
 library.
It communicates via the UCI protocol, so it can be plugged into chess GUIs (like Arena, CuteChess, or Banksia) or run as a backend for a Lichess bot.

The engine combines classic search techniques and heuristics to reach a solid playing strength in a lightweight, easy-to-understand codebase.

⚡ Features

UCI protocol support – works in any GUI or with lichess-bot.

Search:

Iterative deepening

Alpha-beta pruning

Aspiration windows

Quiescence search

Null-move pruning

Late move reductions (LMR)

Heuristics:

Transposition table with replacement scheme

Killer move heuristic

History heuristic

MVV-LVA move ordering

Evaluation:

Material balance

Piece-square tables (PSQT)

Mobility bonus

Passed pawns & pawn structure evaluation

King safety heuristic

Time management – adapts search depth based on available clock time and increments.

Opening book support (via polyglot .bin files).

⚙️ Requirements

Python 3.8+

python-chess (pip install chess)

🚀 Usage

Run directly as a UCI engine:

python3 p-chess_engine.py


Then load it into your favorite GUI or connect it to lichess-bot
.

🎯 Goals

P-chess-Engine is designed to be:

Readable – code is commented and structured for learning.

Portable – no external binaries, just Python + python-chess.

Strong enough for casual play (≈1500–2000 Elo depending on settings).

Customizable – easy to extend with new evaluation features or search tweaks.
