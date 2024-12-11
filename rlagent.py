import pygame
import random
import pickle
import numpy as np
from collections import defaultdict

from matris import Game, WIDTH, HEIGHT, MATRIX_WIDTH, MATRIX_HEIGHT, BLOCKSIZE, list_of_tetrominoes

ACTIONS = [
    pygame.K_LEFT,    # Move left
    pygame.K_RIGHT,   # Move right
    pygame.K_UP,      # Rotate
    pygame.K_SPACE,   # Hard drop
    pygame.K_c        # Hold piece
]

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.95, epsilon=1.0):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(float)  # Q[(state, action)] = value

    def get_state(self, matris):
        # Simplified state representation
        heights = [0] * MATRIX_WIDTH
        for x in range(MATRIX_WIDTH):
            for y in range(MATRIX_HEIGHT):
                cell = matris.matrix[(y, x)]
                if cell is not None and cell[0] == 'block':
                    heights[x] = MATRIX_HEIGHT - y
                    break
        # Compute the number of holes
        n_holes = self.count_holes(matris)
        # Current piece index
        piece_idx = list_of_tetrominoes.index(matris.current_tetromino)
        # Combine features into a tuple
        state = (tuple(heights), n_holes, piece_idx)
        return state

    def count_holes(self, matris):
        n_holes = 0
        for x in range(MATRIX_WIDTH):
            column_filled = False
            for y in range(MATRIX_HEIGHT):
                cell = matris.matrix[(y, x)]
                if cell is not None and cell[0] == 'block':
                    column_filled = True
                elif column_filled:
                    n_holes += 1
        return n_holes

    def choose_action(self, state):
        # Epsilon-greedy
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            # pick action with highest Q-value
            qvals = [self.Q[(state, a_idx)] for a_idx in range(len(self.actions))]
            max_q = max(qvals)
            # Handle multiple actions with the same max Q-value
            best_actions = [a_idx for a_idx, q in enumerate(qvals) if q == max_q]
            best_action = random.choice(best_actions)
            return self.actions[best_action]

    def update(self, old_state, action, reward, new_state):
        action_idx = self.actions.index(action)
        old_q = self.Q[(old_state, action_idx)]
        if new_state is not None:
            next_q = max([self.Q[(new_state, a_idx)] for a_idx in range(len(self.actions))])
        else:
            next_q = 0
        self.Q[(old_state, action_idx)] = old_q + self.alpha * (reward + self.gamma * next_q - old_q)

def run_training(episodes=1000, render=False):
    pygame.init()
    agent = QLearningAgent(actions=ACTIONS, alpha=0.1, gamma=0.95, epsilon=1.0)

    for ep in range(episodes):
        game = Game(render=render)
        done = False
        old_state = agent.get_state(game.matris)

        while not done:
            # Choose action
            action = agent.choose_action(old_state)

            # Take a step
            done, reward, score = game.step(action)

            if not done:
                new_state = agent.get_state(game.matris)
            else:
                new_state = None

            # Update Q-values
            agent.update(old_state, action, reward, new_state)

            # Move to next state
            old_state = new_state

        print(f"Episode {ep+1}/{episodes} finished with score {score}")

        # Decay epsilon
        agent.epsilon = max(0.1, agent.epsilon * 0.995)

    pygame.quit()
    # Optional: Save Q-values to a file
    with open("qvalues.pkl", "wb") as f:
        pickle.dump(dict(agent.Q), f)

if __name__ == '__main__':
    run_training(episodes=1000, render=False)
