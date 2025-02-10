from tkinter import Frame, Label, CENTER
import numpy as np
import random
import pickle
from math import log2
from logic_env_2048 import GameLogic  # Game logic

with open("q_table_2048.pkl", "rb") as f:
    q_table = pickle.load(f)


def get_state_from_matrix(mtx):
    """
    Convert the current board (matrix) into a state vector.
    """
    state_vec = []
    for row in mtx:
        for v in row:
            if v == 0:
                state_vec.append(0)
            else:
                state_vec.append(log2(v))
    return np.array(state_vec, dtype=np.float32)


def map_tile_order(tile, uniq_sorted):
    """
    Map a tile value to its order based on the sorted unique nonzero tiles.
    """
    if tile == 0:
        return 0
    return uniq_sorted.index(tile) + 1


def extract_order_state(state):
    """
    Extract an ordered state representation from the state vector.

    Parameters:
        state: A numpy array representing the state vector.

    Returns:
        A tuple representing the ordered state.
    """
    tiles = state.flatten()
    nonzeros = [tile for tile in tiles if tile != 0]
    uniq_sorted = sorted(set(nonzeros))
    ordered = tuple(0 if tile == 0 else map_tile_order(tile, uniq_sorted) for tile in tiles)
    return ordered


def choose_action_greedy(state_rep, q_table, actions=[0, 1, 2, 3]):
    """
    Choose the best action for the given state representation based on the Q-table.
    If the state is not in the Q-table, randomly choose an action.

    Parameters:
        state_rep: A tuple obtained from extract_order_state.
        q_table: The Q-table dictionary.
        actions: List of possible actions.

    Returns:
        The chosen action as an integer.
    """
    if state_rep in q_table:
        qs = q_table[state_rep]
        best_act = int(np.argmax(qs))
        return best_act
    else:
        return random.choice(actions)


SIZE = 300
GRID_LEN = 3  # 3x3 board
GRID_PADDING = 10

BACKGROUND_COLOR_GAME = "#92877d"
BACKGROUND_COLOR_CELL_EMPTY = "#9e948a"

BACKGROUND_COLOR_DICT = {
    2: "#eee4da", 4: "#ede0c8", 8: "#f2b179", 16: "#f59563",
    32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72", 256: "#edcc61",
    512: "#edc850", 1024: "#edc53f", 2048: "#edc22e", 4096: "#eee4da"
}
CELL_COLOR_DICT = {
    2: "#776e65", 4: "#776e65", 8: "#f9f6f2", 16: "#f9f6f2",
    32: "#f9f6f2", 64: "#f9f6f2", 128: "#f9f6f2", 256: "#f9f6f2",
    512: "#f9f6f2", 1024: "#f9f6f2", 2048: "#f9f6f2", 4096: "#776e65"
}
FONT = ("Verdana", 40, "bold")


# Auto-run GUI
class AutoPlayGameGrid(Frame):
    def __init__(self):
        Frame.__init__(self)
        self.grid()
        self.master.title('2048 - Auto Play (3x3)')
        self.grid_cells = []
        self.init_grid()

        # Initialize the board
        self.matrix = GameLogic.new_game(GRID_LEN)
        self.update_grid_cells()

        # Map action numbers to GameLogic functions
        self.action_map = {
            0: GameLogic.up,
            1: GameLogic.down,
            2: GameLogic.left,
            3: GameLogic.right
        }
        # Auto-run
        self.after(500, self.auto_play)

    def init_grid(self):
        """Initialize the display area for the 3x3 board."""
        bg = Frame(self, bg=BACKGROUND_COLOR_GAME, width=SIZE, height=SIZE)
        bg.grid()
        for i in range(GRID_LEN):
            row_cells = []
            for j in range(GRID_LEN):
                cell = Frame(bg, bg=BACKGROUND_COLOR_CELL_EMPTY,
                             width=SIZE / GRID_LEN, height=SIZE / GRID_LEN)
                cell.grid(row=i, column=j, padx=GRID_PADDING, pady=GRID_PADDING)
                lbl = Label(master=cell, text="",
                            bg=BACKGROUND_COLOR_CELL_EMPTY,
                            justify=CENTER, font=FONT, width=4, height=2)
                lbl.grid()
                row_cells.append(lbl)
            self.grid_cells.append(row_cells)

    def update_grid_cells(self):
        """Refresh the 3x3 board display."""
        for i in range(GRID_LEN):
            for j in range(GRID_LEN):
                val = self.matrix[i][j]
                if val == 0:
                    self.grid_cells[i][j].configure(text="", bg=BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(
                        text=str(val),
                        bg=BACKGROUND_COLOR_DICT.get(val, "#3c3a32"),
                        fg=CELL_COLOR_DICT.get(val, "#f9f6f2")
                    )
        self.update_idletasks()

    def auto_play(self):
        """Automatically play the game using the trained Q-table."""
        state_vec = get_state_from_matrix(self.matrix)
        state_rep = extract_order_state(state_vec)
        act = choose_action_greedy(state_rep, q_table)
        mv_fn = self.action_map[act]

        new_mtx, moved = mv_fn(self.matrix)
        if moved:
            # If the move is valid, add a new number and update the display
            self.matrix = GameLogic.add_two(new_mtx)
            self.update_grid_cells()
        else:
            # If the move is invalid, randomly try another move
            act = random.choice([0, 1, 2, 3])
            mv_fn = self.action_map[act]
            new_mtx, moved = mv_fn(self.matrix)
            if moved:
                self.matrix = GameLogic.add_two(new_mtx)
                self.update_grid_cells()

        # Check the game state
        status = GameLogic.game_state(self.matrix)
        if status == 'win':
            self._show_message("You", "Win!")
            return
        elif status == 'lose':
            self._show_message("You", "Lose!")
            return

        # Schedule the next auto-play step after 300 milliseconds
        self.after(300, self.auto_play)

    def _show_message(self, title, message):
        """Display a message on all cells."""
        for i in range(GRID_LEN):
            for j in range(GRID_LEN):
                self.grid_cells[i][j].configure(text=message)


if __name__ == "__main__":
    gamegrid = AutoPlayGameGrid()
    gamegrid.mainloop()
