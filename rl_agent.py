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
    def __init__(self, actions, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(float)  # Q[(state, action)] = value

    def get_state(self, matris):
        # Extract a simple state representation
        # Flatten the board: 1 if filled ('block'), else 0
        board = []
        for y in range(MATRIX_HEIGHT):
            for x in range(MATRIX_WIDTH):
                cell = matris.matrix[(y,x)]
                board.append(1 if cell and cell[0] == 'block' else 0)

        # Current piece index
        piece_idx = list_of_tetrominoes.index(matris.current_tetromino)
        posY, posX = matris.tetromino_position
        rot = matris.tetromino_rotation

        # State as a tuple
        state = tuple(board) + (piece_idx, posY, posX, rot)
        return state

    def choose_action(self, state):
        # Epsilon-greedy
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            # pick action with highest Q-value
            qvals = [self.Q[(state, a)] for a in range(len(self.actions))]
            best_action = np.argmax(qvals)
            return self.actions[best_action]

    def update(self, old_state, action, reward, new_state):
        old_q = self.Q[(old_state, self.actions.index(action))]
        next_q = max([self.Q[(new_state, a)] for a in range(len(self.actions))]) if not (new_state is None) else 0
        self.Q[(old_state, self.actions.index(action))] = old_q + self.alpha * (reward + self.gamma * next_q - old_q)


def run_training(episodes=10, render=False):
    """
    Runs several episodes of Tetris using Q-learning. 
    """
    pygame.init()
    agent = QLearningAgent(actions=ACTIONS, alpha=0.1, gamma=0.95, epsilon=0.1)

    for ep in range(episodes):
        game = Game(render=render)
        done = False
        final_score = 0

        # Get initial state
        old_state = agent.get_state(game.matris)
        old_score = game.matris.score

        while not done:
            # Choose action
            action = agent.choose_action(old_state)

            # Take a step
            done, score = game.step(action)
            reward = score - old_score
            old_score = score

            if not done:
                new_state = agent.get_state(game.matris)
            else:
                new_state = None

            # Update Q-values
            agent.update(old_state, action, reward, new_state)

            # Move to next state
            old_state = new_state

        print(f"Episode {ep+1}/{episodes} finished with score {final_score}")

    pygame.quit()
    # Optional: Save Q-values to a file
    # with open("qvalues.pkl", "wb") as f:
    #     pickle.dump(agent.Q, f)


if __name__ == '__main__':
    run_training(episodes=5, render=True)
