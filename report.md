# Q‑Learning Training Report for the 2048 Game

This report describes the overall design and implementation of the experiment. It covers the game design, the Q‑learning setup (including state space, action space, reward function, exploration strategy, and parameter settings), and a final summary of the experimental findings.

## 1. Game Design

We use the 2048 game as our research subject, but to reduce the state space and speed up training, we shrink the board to a 3×3 version. At the start of the game, two tiles with the value 2 are placed randomly on the board. The player (or agent) can move tiles up, down, left, or right. Whenever a move is made, all non-empty tiles slide in that direction; adjacent tiles of the same value merge into a single tile with a larger value. The goal is to merge tiles to form higher-valued numbers. This design retains the core merging mechanism of 2048 while greatly reducing the number of possible states, making it easier for the Q‑learning algorithm to converge faster.

## 2. Q‑Learning Implementation

### 2.1 State Space and Action Space

In this experiment, the action space is fixed at four actions, represented by the numbers 0, 1, 2, and 3 for moving up, down, left, and right, respectively.

For the state space, we first convert each 3×3 board tile into a numerical value: if a tile is empty, it is labeled 0; if it is not empty, we take the log base 2 of that tile’s value. Next, we gather all non-zero values and sort them. We assign them ranks starting from 1, with tiles of equal value getting the same rank, while empty spaces remain 0. This approach preserves only the relative ordering of tile values (e.g., largest tile, second largest tile) instead of absolute values, which significantly reduces the total number of states and helps the agent learn more quickly.

### 2.2 Reward Function

The reward function in the code is composed of several parts:

- **Environment Reward**: This comes directly from the game’s merging score. The reward is normalized before being used.
- **Extra Reward**: This is based on the change in the number of empty cells. Empty cells are labeled 0 in the state representation. If the number of empty cells increases after a move, the agent receives a positive reward; if it decreases, the agent receives a negative reward. This reward is also normalized.
- **Merging Reward**: Whenever a new maximum tile is generated, we calculate the difference between the new and old max tile and award bonus points according to a preset weight. Later adjustments increased the reward for creating a new highest tile, thus encouraging the agent to produce larger values.
- **Penalties**: If the board does not change after a move, we regard that move as invalid and subtract some points. If the game ends with a negative environment reward, we also apply an extra penalty for losing.

All of these rewards and penalties are normalized to keep their values in a reasonable range, so that the agent can distinguish between good moves and bad moves and learn appropriate strategies.

### 2.3 Algorithm Parameters and Exploration Strategy

We use an epsilon‑greedy strategy in Q‑learning. Initially, the agent acts almost entirely at random (epsilon = 1.0) and gradually decays epsilon to a minimum value of 0.1, thereby reducing randomness while increasingly relying on the knowledge gained. In addition, the learning rate was set higher at later stages to allow the Q values to adjust more rapidly, and we increased the discount factor so that the agent places more importance on future rewards. Through iterative testing and tuning of the learning rate and discount factor, we help the agent converge more quickly.

## 3. Experimental Summary

After extensive training, the agent learned to merge larger tiles and effectively gain immediate rewards. However, the agent did not learn to arrange the board in a neat order that might enable better long-term merges. One likely reason is that our current state representation focuses only on relative tile ranks and ignores the broader board structure. At the same time, the reward function heavily favors immediate merges, offering insufficient encouragement for keeping the board orderly. As a result, the agent centers its strategy on combining larger tiles quickly rather than maintaining a layout conducive to future merges.

In conclusion, using a 3×3 board and rank-based state compression greatly reduces the state space and speeds up training. Our reward function includes normalized environment scores, extra rewards for empty cells, and special bonuses for generating higher-value tiles. We use epsilon‑greedy for exploration and set relatively high learning-rate and discount-factor values so the agent can quickly move toward an optimal policy. Although the agent performs reasonably well with immediate merges, it does not learn more advanced strategies to keep the board tidy. Future improvements could include adding more board-structure details to the state representation and providing additional rewards for maintaining an orderly layout, which would encourage longer-term planning. This experiment provides useful insights and ideas for further work on applying Q‑learning to the 2048 game.


Github：[xiaofeifei111/q-leaning-2048: q-leaning-2048](https://github.com/xiaofeifei111/q-leaning-2048)