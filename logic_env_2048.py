import random
import numpy as np


class GameLogic:
    @staticmethod
    def new_game(n):
        """Initialize a new game: create an n x n grid and add two 2's."""
        mtx = [[0] * n for _ in range(n)]
        mtx = GameLogic.add_two(mtx)
        mtx = GameLogic.add_two(mtx)
        return mtx

    @staticmethod
    def add_two(mtx):
        """Add a 2 at a random empty cell."""
        r = random.randint(0, len(mtx) - 1)
        c = random.randint(0, len(mtx) - 1)
        while mtx[r][c] != 0:
            r = random.randint(0, len(mtx) - 1)
            c = random.randint(0, len(mtx) - 1)
        mtx[r][c] = 2
        return mtx

    @staticmethod
    def game_state(mtx):
        """
        Check the current game state.
        Return 'win' if 2048 is reached,
        'not over' if moves are still possible,
        otherwise return 'lose'.
        """
        # Check if 2048 is reached.
        for r in range(len(mtx)):
            for c in range(len(mtx[0])):
                if mtx[r][c] == 2048:
                    return 'win'

        # Check for any empty cell.
        for r in range(len(mtx)):
            for c in range(len(mtx[0])):
                if mtx[r][c] == 0:
                    return 'not over'

        # Check if adjacent cells can merge.
        for r in range(len(mtx) - 1):
            for c in range(len(mtx[0]) - 1):
                if mtx[r][c] == mtx[r + 1][c] or mtx[r][c] == mtx[r][c + 1]:
                    return 'not over'
        # Check the last row.
        n = len(mtx)
        for k in range(n - 1):
            if mtx[n - 1][k] == mtx[n - 1][k + 1]:
                return 'not over'
        # Check the last column.
        for r in range(n - 1):
            if mtx[r][n - 1] == mtx[r + 1][n - 1]:
                return 'not over'

        return 'lose'

    @staticmethod
    def reverse(mtx):
        """Flip the grid horizontally."""
        return [row[::-1] for row in mtx]

    @staticmethod
    def transpose(mtx):
        """Transpose the grid."""
        return [[mtx[r][c] for r in range(len(mtx))] for c in range(len(mtx[0]))]

    @staticmethod
    def cover_up(mtx):
        """
        Compress the grid to the left.
        Returns a tuple (new_mtx, moved) where moved is True if any cell shifted.
        """
        rows = len(mtx)
        cols = len(mtx[0])
        new_mtx = [[0] * cols for _ in range(rows)]
        moved = False
        for r in range(rows):
            pos = 0
            for c in range(cols):
                if mtx[r][c] != 0:
                    new_mtx[r][pos] = mtx[r][c]
                    if c != pos:
                        moved = True
                    pos += 1
        return new_mtx, moved

    @staticmethod
    def merge(mtx, moved):
        """
        Merge adjacent identical cells to the left.
        Returns a tuple (new_mtx, moved) where moved is True if any merge occurred.
        """
        rows = len(mtx)
        cols = len(mtx[0])
        for r in range(rows):
            for c in range(cols - 1):
                if mtx[r][c] == mtx[r][c + 1] and mtx[r][c] != 0:
                    mtx[r][c] *= 2
                    mtx[r][c + 1] = 0
                    moved = True
        return mtx, moved

    @staticmethod
    def up(mtx):
        """Move up."""
        mtx = GameLogic.transpose(mtx)
        mtx, moved = GameLogic.cover_up(mtx)
        mtx, moved = GameLogic.merge(mtx, moved)
        mtx, _ = GameLogic.cover_up(mtx)
        mtx = GameLogic.transpose(mtx)
        return mtx, moved

    @staticmethod
    def down(mtx):
        """Move down."""
        mtx = GameLogic.reverse(GameLogic.transpose(mtx))
        mtx, moved = GameLogic.cover_up(mtx)
        mtx, moved = GameLogic.merge(mtx, moved)
        mtx, _ = GameLogic.cover_up(mtx)
        mtx = GameLogic.transpose(GameLogic.reverse(mtx))
        return mtx, moved

    @staticmethod
    def left(mtx):
        """Move left."""
        mtx, moved = GameLogic.cover_up(mtx)
        mtx, moved = GameLogic.merge(mtx, moved)
        mtx, _ = GameLogic.cover_up(mtx)
        return mtx, moved

    @staticmethod
    def right(mtx):
        """Move right."""
        mtx = GameLogic.reverse(mtx)
        mtx, moved = GameLogic.cover_up(mtx)
        mtx, moved = GameLogic.merge(mtx, moved)
        mtx, _ = GameLogic.cover_up(mtx)
        mtx = GameLogic.reverse(mtx)
        return mtx, moved


class Game2048Env:
    def __init__(self, size=4):
        self.size = size
        self.matrix = None
        self.done = False
        self.score = 0
        # Action mapping: 0=up, 1=down, 2=left, 3=right
        self._action_map = {
            0: GameLogic.up,
            1: GameLogic.down,
            2: GameLogic.left,
            3: GameLogic.right
        }

    def reset(self):
        """Reset the game and return the initial state."""
        self.matrix = GameLogic.new_game(self.size)
        self.done = False
        self.score = 0
        return self._get_state()

    def step(self, action):
        """
        Perform an action.
        action: 0=up, 1=down, 2=left, 3=right
        """
        if self.done:
            # If the game is over, further steps do not update the state.
            return self._get_state(), 0, True, {}

        old_sum = self._sum_matrix(self.matrix)

        # Execute the move.
        move_fn = self._action_map[action]
        new_matrix, moved = move_fn(self.matrix)
        self.matrix = new_matrix

        # If a valid move occurred, add a new '2' in an empty cell.
        if moved:
            self.matrix = GameLogic.add_two(self.matrix)

        # Calculate the increment based on the change in total sum.
        new_sum = self._sum_matrix(self.matrix)
        inc = new_sum - old_sum
        self.score += inc  # Update score.

        # Check the game state.
        gs = GameLogic.game_state(self.matrix)
        if gs == 'win':
            self.done = True
            reward = 100 + inc
        elif gs == 'lose':
            self.done = True
            reward = -10  # Penalty for losing.
        else:
            self.done = False
            reward = inc

        return self._get_state(), reward, self.done, {}

    def render(self):
        """Print the grid and current score."""
        print("Current grid:")
        for row in self.matrix:
            print(row)
        print("Score:", self.score)

    @property
    def action_space(self):
        return [0, 1, 2, 3]

    def _sum_matrix(self, mtx):
        """Return the sum of all values in the grid."""
        return sum(sum(row) for row in mtx)

    def _get_state(self):
        """
        Convert the 4x4 grid into a 16-dimensional vector.
        For non-zero cells, take log2 of the value.
        """
        state = []
        for row in self.matrix:
            for v in row:
                state.append(0 if v == 0 else np.log2(v))
        return np.array(state, dtype=np.float32)
