import random
import pickle
import numpy as np
from logic_env_2048 import Game2048Env

def map_tile_order(tile, unique_sorted):
    if tile == 0:
        return 0
    return unique_sorted.index(tile) + 1

def extract_order_state(state):
    flat = state.flatten()
    nonzero = [tile for tile in flat if tile != 0]
    unique_sorted = sorted(set(nonzero))
    ordered = tuple(0 if tile == 0 else map_tile_order(tile, unique_sorted) for tile in flat)
    return ordered

class QLearningAgent:
    def __init__(self, actions, alpha=0.6, gamma=0.8,
                 epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = {}

    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        return self.q_table[state]

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            q_values = self.get_q_values(state)
            max_actions = np.where(q_values == q_values.max())[0]
            return int(random.choice(max_actions))

    def update(self, state, action, reward, next_state, done):
        q_values = self.get_q_values(state)
        if done:
            target = reward
        else:
            next_q = self.get_q_values(next_state)
            target = reward + self.gamma * np.max(next_q)
        q_values[action] += self.alpha * (target - q_values[action])

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


ENV_SCALE = 10
EXTRA_SCALE = 1.0
MERGE_SCALE = 100.0

INVALID_MOVE_PENALTY = -5.0
FAILURE_PENALTY = -50.0

def compute_extra_reward_new(state, next_state, weights):
    empty_old = sum(1 for d in state if d == 0)
    empty_new = sum(1 for d in next_state if d == 0)
    raw_empty_reward = weights.get('empty', 0) * (empty_new - empty_old)
    norm_empty_reward = raw_empty_reward / EXTRA_SCALE
    return norm_empty_reward

# train
def train_q_learning(episodes=10000):
    env = Game2048Env(size=3)
    agent = QLearningAgent(actions=env.action_space,
                           alpha=0.7, gamma=0.9,
                           epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.998)
    reward_weights = {'empty': 1.0, 'max': 10.0}
    total_rewards = []
    max_tiles = []

    for ep in range(episodes):
        state = env.reset()
        order_state = extract_order_state(state)
        ep_reward = 0.0
        step_count = 0
        done = False
        current_max_tile = 2 ** np.max(state)
        episode_max_tile = current_max_tile

        while not done:
            step_count += 1
            action = agent.choose_action(order_state)
            next_state, reward_env, done, _ = env.step(action)
            new_order_state = extract_order_state(next_state)

            raw_extra_reward = compute_extra_reward_new(order_state, new_order_state,
                                                        weights={'empty': reward_weights['empty']})
            norm_env_reward = reward_env / ENV_SCALE

            old_max_tile = 2 ** np.max(state)
            new_max_tile = 2 ** np.max(next_state)
            raw_merge_reward = 0.0
            if new_max_tile > old_max_tile:
                extra_factor = np.log2(new_max_tile)
                raw_merge_reward = reward_weights['max'] * (new_max_tile - old_max_tile) * extra_factor
            norm_merge_reward = raw_merge_reward / MERGE_SCALE

            invalid_penalty = 0.0
            if np.array_equal(next_state, state):
                invalid_penalty = INVALID_MOVE_PENALTY / ENV_SCALE

            failure_penalty = 0.0
            if done and reward_env < 0:
                failure_penalty = FAILURE_PENALTY / ENV_SCALE

            total_reward = norm_env_reward + raw_extra_reward + norm_merge_reward + invalid_penalty + failure_penalty
            ep_reward += total_reward
            episode_max_tile = max(episode_max_tile, new_max_tile)


            agent.update(order_state, action, total_reward, new_order_state, done)
            order_state = new_order_state
            state = next_state

        agent.decay_epsilon()
        total_rewards.append(ep_reward)
        max_tiles.append(episode_max_tile)
        if (ep + 1) % 100 == 0:
            avg_reward = sum(total_rewards[-100:]) / 100
            avg_max_tile = sum(max_tiles[-100:]) / 100
            print(f"Episode {ep + 1:5d}, Average Reward (last 100): {avg_reward:7.2f}, "
                  f"Avg Max Tile: {avg_max_tile:7.2f}, Epsilon: {agent.epsilon:5.3f}")
            print(f"After {ep + 1} episodes, Q-table contains {len(agent.q_table)} states.")

    return agent


if __name__ == "__main__":
    num_episodes = 100000
    trained_agent = train_q_learning(episodes=num_episodes)
    with open("q_table_2048.pkl", "wb") as f:
        pickle.dump(trained_agent.q_table, f)
    print("Training completed")
