import tkinter as tk
from tkinter import Frame, Label, CENTER
import numpy as np
import random
from logic_env_2048 import Game2048Env


class QLearningAgent:
    def __init__(self, actions, alpha=0.5, gamma=0.9,
                 epsilon=1.0, epsilon_min=0.2, epsilon_decay=0.999):
        """
        actions: list of available actions, e.g., [0, 1, 2, 3]
        """
        self.actions = actions
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = {}  # Q-table

    def get_q_values(self, state):
        """Initialize Q-values for unseen states to 0 for all actions."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        return self.q_table[state]

    def choose_action(self, state):
        """Select an action using the epsilon-greedy strategy."""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            qs = self.get_q_values(state)
            max_acts = np.where(qs == qs.max())[0]
            return int(random.choice(max_acts))

    def update(self, state, action, reward, next_state, done):
        """Update the Q-table using the Q-learning update rule."""
        qs = self.get_q_values(state)
        if done:
            target = reward
        else:
            next_qs = self.get_q_values(next_state)
            target = reward + self.gamma * np.max(next_qs)
        qs[action] += self.alpha * (target - qs[action])

    def decay_epsilon(self):
        """Decay the epsilon value after each episode."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def extract_features(state):
    """
    Input: state is a numpy array of shape (9,) where each non-zero element is log2(tile)
    Output:
       - features: (mono, smooth_bin, empty, max_pos)
           mono: monotonicity score (for rows and columns, range 0–6)
           smooth_bin: discretized smoothness value
           empty: number of empty cells
           max_pos: category of the max tile's position
       - raw smooth: raw smoothness value
    """
    board = state.reshape(3, 3)

    # Calculate monotonicity
    mono = 0
    # Rows
    for row in board:
        if (all(row[i] <= row[i + 1] for i in range(2)) or
            all(row[i] >= row[i + 1] for i in range(2))):
            mono += 1
    # Columns
    for j in range(3):
        col = board[:, j]
        if (all(col[i] <= col[i + 1] for i in range(2)) or
            all(col[i] >= col[i + 1] for i in range(2))):
            mono += 1

    # Calculate smoothness
    smooth = 0.0
    for i in range(3):
        for j in range(3):
            if j < 2:  # right neighbor
                smooth += (board[i, j] - board[i, j + 1]) ** 2
            if i < 2:  # bottom neighbor
                smooth += (board[i, j] - board[i + 1, j]) ** 2
    # Discretize smoothness
    bin_size = 5.0
    smooth_bin = int(smooth / bin_size)

    # Count empty cells
    empty = int(np.sum(board == 0))

    # Determine max tile position category
    max_pos_arr = np.argwhere(board == np.max(board))
    pos = max_pos_arr[0]  # take the first occurrence (row, col)
    # For 3x3: (0,0), (0,2), (2,0), (2,2) are corners;
    # (0,1), (1,0), (1,2), (2,1) are edges;
    # (1,1) is center.
    if (pos[0] in [0, 2]) and (pos[1] in [0, 2]):
        max_pos = 0  # corner
    elif (pos[0] in [0, 2]) or (pos[1] in [0, 2]):
        max_pos = 1  # edge
    else:
        max_pos = 2  # center

    feats = (mono, smooth_bin, empty, max_pos)
    return feats, smooth


def compute_extra_reward(old_feats, new_feats, old_smooth, new_smooth, weights):
    extra = 0.0
    # Reward for empty cells
    extra += weights['empty'] * (new_feats[2] - old_feats[2])
    # Reward for monotonicity
    extra += weights['mono'] * (new_feats[0] - old_feats[0])
    # Reward for smoothness improvement
    extra += weights['smooth'] * (old_smooth - new_smooth)
    # Bonus if the max tile is in a corner
    if new_feats[3] == 0:
        extra += 2.0
    return extra


class TrainVisualization3x3(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()

        # Initialize the board display
        self.init_grid()
        self.init_info_labels()

        self.env = Game2048Env(size=3)
        # Create the Q-learning agent
        self.agent = QLearningAgent(
            actions=self.env.action_space,
            alpha=0.5, gamma=0.9,
            epsilon=1.0, epsilon_min=0.2, epsilon_decay=0.999
        )

        # Record training information
        self.episode = 1
        self.step_count = 0
        self.cumulative_reward = 0.0

        # Start the first episode
        self.new_episode()

    def new_episode(self):
        """Reset the game environment and start a new training episode."""
        self.state = self.env.reset()
        self.current_features, self.old_smooth = extract_features(self.state)
        self.cumulative_reward = 0.0
        self.step_count = 0
        self.update_grid_cells()
        self.update_info_labels(step_reward=0)
        # Schedule the training loop after a delay
        self.after(1000, self.train_step)

    def train_step(self):
        """Execute a training step."""
        if self.env.done:
            self.agent.decay_epsilon()
            print(f"Episode {self.episode} finished with cumulative reward {self.cumulative_reward:.2f}")
            self.episode += 1
            self.after(1000, self.new_episode)
            return

        # Choose an action based on current features
        act = self.agent.choose_action(self.current_features)
        # Execute the action
        nxt_state, rew_env, done, _ = self.env.step(act)
        new_feats, new_smooth = extract_features(nxt_state)

        # Compute extra reward from features
        extra_rew = compute_extra_reward(
            self.current_features, new_feats,
            self.old_smooth, new_smooth,
            weights={'empty': 1.0, 'mono': 1.0, 'smooth': 0.5, 'max': 2.0}
        )
        # Bonus reward for merging to form a higher tile
        old_max = 2 ** np.max(self.state)  # state stores log2 values
        new_max = 2 ** np.max(nxt_state)
        extra_merge = 0.0
        if new_max > old_max:
            extra_merge = 2.0 * (new_max - old_max)

        # Total reward for the step
        tot_rew = rew_env + extra_rew + extra_merge

        # Q-learning update
        self.agent.update(self.current_features, act, tot_rew, new_feats, done)

        # Transition to the next state
        self.state = nxt_state
        self.current_features = new_feats
        self.old_smooth = new_smooth

        # Update display information
        self.cumulative_reward += tot_rew
        self.step_count += 1
        self.update_grid_cells()
        self.update_info_labels(step_reward=tot_rew)

        # Schedule the next training step
        if not done:
            self.after(1000, self.train_step)
        else:
            # If episode has ended, schedule next step to initiate new episode
            self.after(1000, self.train_step)

    def init_grid(self):
        """Initialize the display area for the 3x3 board."""
        self.grid_cells = []
        bg = Frame(self, bg="#92877d", width=300, height=300)
        bg.grid(row=0, column=0)
        for i in range(3):
            row = []
            for j in range(3):
                cell = Frame(bg, bg="#9e948a", width=100, height=100)
                cell.grid(row=i, column=j, padx=10, pady=10)
                lbl = Label(cell, text="", bg="#9e948a", justify=CENTER,
                            font=("Verdana", 30, "bold"), width=3, height=1)
                lbl.grid()
                row.append(lbl)
            self.grid_cells.append(row)

    def update_grid_cells(self):
        """Refresh the 3x3 board display."""
        # self.env.matrix should be the actual (3,3) grid values
        for i in range(3):
            for j in range(3):
                val = self.env.matrix[i][j]
                if val == 0:
                    self.grid_cells[i][j].configure(text="", bg="#9e948a")
                else:
                    self.grid_cells[i][j].configure(text=str(val), bg="#eee4da", fg="#776e65")
        self.update_idletasks()

    def init_info_labels(self):
        """Initialize the information display: episode, step count, and rewards."""
        self.info_frame = Frame(self)
        self.info_frame.grid(row=1, column=0)
        self.episode_label = Label(self.info_frame, text="Episode: 1")
        self.episode_label.pack(side="left", padx=10)
        self.step_label = Label(self.info_frame, text="Step: 0")
        self.step_label.pack(side="left", padx=10)
        self.step_reward_label = Label(self.info_frame, text="Step Reward: 0")
        self.step_reward_label.pack(side="left", padx=10)
        self.cumulative_label = Label(self.info_frame, text="Cumulative Reward: 0")
        self.cumulative_label.pack(side="left", padx=10)

    def update_info_labels(self, step_reward=0):
        """Update the display for episode, step count, step reward, and cumulative reward."""
        self.episode_label.config(text=f"Episode: {self.episode}")
        self.step_label.config(text=f"Step: {self.step_count}")
        self.step_reward_label.config(text=f"Step Reward: {step_reward:.2f}")
        self.cumulative_label.config(text=f"Cumulative Reward: {self.cumulative_reward:.2f}")


if __name__ == "__main__":
    root = tk.Tk()
    root.title("2048")
    tv = TrainVisualization3x3(root)
    tv.pack()
    root.mainloop()
