import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from matris import Game, WIDTH, HEIGHT, MATRIX_WIDTH, MATRIX_HEIGHT, BLOCKSIZE, list_of_tetrominoes

# Define possible actions
ACTIONS = [
    pygame.K_LEFT,    # Move left
    pygame.K_RIGHT,   # Move right
    pygame.K_UP,      # Rotate
    pygame.K_SPACE,   # Hard drop
    pygame.K_c        # Hold piece
]

# Hyperparameters
STATE_SIZE = MATRIX_WIDTH + 4  # Heights + aggregate_height + num_holes + bumpiness + lines_cleared
ACTION_SIZE = len(ACTIONS)
BATCH_SIZE = 64
GAMMA = 0.99
LR = 0.001
MEMORY_SIZE = 10000
TARGET_UPDATE = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

class Agent:
    def __init__(self):
        self.policy_net = DQN(STATE_SIZE, ACTION_SIZE).to(device)
        self.target_net = DQN(STATE_SIZE, ACTION_SIZE).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory = deque(maxlen=MEMORY_SIZE)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.criterion = nn.MSELoss()

        self.steps_done = 0
        self.epsilon = 1.0  # Start with full exploration

    def select_action(self, state):
        sample = random.random()
        epsilon_threshold = self.epsilon
        if sample > epsilon_threshold:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).to(device)
                q_values = self.policy_net(state)
                action_index = torch.argmax(q_values).item()
                return ACTIONS[action_index]
        else:
            return random.choice(ACTIONS)

    def store_transition(self, state, action, reward, next_state, done):
        action_index = ACTIONS.index(action)
        self.memory.append((state, action_index, reward, next_state, done))

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = random.sample(self.memory, BATCH_SIZE)
        batch = list(zip(*transitions))

        state_batch = torch.tensor(np.array(batch[0]), dtype=torch.float32).to(device)
        action_batch = torch.tensor(batch[1]).unsqueeze(1).to(device)
        reward_batch = torch.tensor(batch[2], dtype=torch.float32).to(device)
        next_state_batch = torch.tensor(np.array(batch[3]), dtype=torch.float32).to(device)
        done_batch = torch.tensor(batch[4], dtype=torch.float32).to(device)

        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0]
            next_state_values[done_batch == True] = 0.0  # If done, no future reward

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute loss
        loss = self.criterion(state_action_values.squeeze(), expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def run_training(episodes=1000, render=False):
    pygame.init()
    env = Game(render=render)
    agent = Agent()

    for ep in range(episodes):
        env.reset()
        state = env.matris.get_state_features()
        total_reward = 0
        done = False

        while not done:
            # Select and perform an action
            action = agent.select_action(state)
            next_state, reward, done, score = env.step(action)
            total_reward += reward;

            # Store the transition in memory
            agent.store_transition(state, action, reward, next_state, done)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization
            agent.optimize_model()

        # Update the target network
        if ep % TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        # Decay epsilon
        agent.epsilon = max(0.1, agent.epsilon * 0.995)

        print(f"Episode {ep+1}/{episodes}, Score: {score}, Total Reward: {total_reward}")

    pygame.quit()
    # Optional: Save the trained model
    torch.save(agent.policy_net.state_dict(), "tetris_dqn.pth")

if __name__ == '__main__':
    run_training(episodes=1000, render=True)
