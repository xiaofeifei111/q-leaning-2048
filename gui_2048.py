from tkinter import Frame, Label, CENTER
from logic_env_2048 import GameLogic

SIZE = 400
GRID_LEN = 4
GRID_PADDING = 10

BACKGROUND_COLOR_GAME = "#92877d"
BACKGROUND_COLOR_CELL_EMPTY = "#9e948a"

BACKGROUND_COLOR_DICT = {
    2: "#eee4da", 4: "#ede0c8", 8: "#f2b179", 16: "#f59563",
    32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72", 256: "#edcc61",
    512: "#edc850", 1024: "#edc53f", 2048: "#edc22e", 4096: "#eee4da",
    8192: "#edc22e", 16384: "#f2b179", 32768: "#f59563", 65536: "#f67c5f"
}

CELL_COLOR_DICT = {
    2: "#776e65", 4: "#776e65", 8: "#f9f6f2", 16: "#f9f6f2",
    32: "#f9f6f2", 64: "#f9f6f2", 128: "#f9f6f2", 256: "#f9f6f2",
    512: "#f9f6f2", 1024: "#f9f6f2", 2048: "#f9f6f2", 4096: "#776e65",
    8192: "#f9f6f2", 16384: "#776e65", 32768: "#776e65", 65536: "#f9f6f2"
}

FONT = ("Verdana", 40, "bold")

KEY_QUIT = "Escape"
KEY_BACK = "b"
KEY_UP = "Up"
KEY_DOWN = "Down"
KEY_LEFT = "Left"
KEY_RIGHT = "Right"


class GameGrid(Frame):
    def __init__(self):
        Frame.__init__(self)

        self.grid()
        self.master.title('2048')
        self.master.bind("<Key>", self.key_down)

        # 与 GameLogic 中的 up/down/left/right 对应
        self.commands = {
            KEY_UP: GameLogic.up,
            KEY_DOWN: GameLogic.down,
            KEY_LEFT: GameLogic.left,
            KEY_RIGHT: GameLogic.right
        }

        self.grid_cells = []
        self.init_grid()

        # 初始化矩阵
        self.matrix = GameLogic.new_game(GRID_LEN)
        # 存历史用于回退
        self.history_matrixs = []

        self.update_grid_cells()
        self.mainloop()

    def init_grid(self):
        background = Frame(self, bg=BACKGROUND_COLOR_GAME, width=SIZE, height=SIZE)
        background.grid()
        for i in range(GRID_LEN):
            grid_row = []
            for j in range(GRID_LEN):
                cell = Frame(
                    background,
                    bg=BACKGROUND_COLOR_CELL_EMPTY,
                    width=SIZE / GRID_LEN,
                    height=SIZE / GRID_LEN
                )
                cell.grid(row=i, column=j, padx=GRID_PADDING, pady=GRID_PADDING)
                t = Label(
                    master=cell,
                    text="",
                    bg=BACKGROUND_COLOR_CELL_EMPTY,
                    justify=CENTER,
                    font=FONT,
                    width=5,
                    height=2
                )
                t.grid()
                grid_row.append(t)
            self.grid_cells.append(grid_row)

    def update_grid_cells(self):
        for i in range(GRID_LEN):
            for j in range(GRID_LEN):
                new_number = self.matrix[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(
                        text="",
                        bg=BACKGROUND_COLOR_CELL_EMPTY
                    )
                else:
                    self.grid_cells[i][j].configure(
                        text=str(new_number),
                        bg=BACKGROUND_COLOR_DICT.get(new_number, "#3c3a32"),
                        fg=CELL_COLOR_DICT.get(new_number, "#f9f6f2")
                    )
        self.update_idletasks()

    def key_down(self, event):
        key = event.keysym
        if key == KEY_QUIT:
            exit()
        if key == KEY_BACK and len(self.history_matrixs) > 1:
            self.matrix = self.history_matrixs.pop()
            self.update_grid_cells()
        elif key in self.commands:
            # 记录旧状态
            self.history_matrixs.append([row[:] for row in self.matrix])

            # 执行动作
            self.matrix, done = self.commands[key](self.matrix)
            if done:
                # 若有移动，添加新的 '2'
                self.matrix = GameLogic.add_two(self.matrix)
                self.update_grid_cells()
                # 检查游戏状态
                game_state = GameLogic.game_state(self.matrix)
                if game_state == 'win':
                    self._show_message("You", "Win!")
                elif game_state == 'lose':
                    self._show_message("You", "Lose!")

    def _show_message(self, msg1, msg2):
        self.grid_cells[1][1].configure(text=msg1)
        self.grid_cells[1][2].configure(text=msg2)


if __name__ == "__main__":
    GameGrid()
