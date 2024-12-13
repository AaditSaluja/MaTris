import pygame
import random
import numpy as np
from matris import Game, MATRIX_WIDTH, MATRIX_HEIGHT, BLOCKSIZE, list_of_tetrominoes, GameOver

ACTIONS = [
    pygame.K_LEFT,    # Move left
    pygame.K_RIGHT,   # Move right
    pygame.K_UP,      # Rotate
    pygame.K_SPACE,   # Hard drop
    pygame.K_c        # Hold piece
]

class PolicyIterationAgent:
    def __init__(self, gamma=0.99, theta=1e-6):
        self.gamma = gamma  # Discount factor
        self.theta = theta  # Convergence threshold
        self.policy = {}  # Maps states (hashed feature vectors) to actions
        self.value_function = {}  # Maps states (hashed feature vectors) to values

    def policy_evaluation(self, environment):
        '''Evaluate the current policy to calculate the value function.'''
        while True:
            delta = 0
            for state_key, action in self.policy.items():
                transitions = environment.get_transitions(state_key, action)
                v = self.value_function.get(state_key, 0)
                self.value_function[state_key] = sum(
                    prob * (reward + self.gamma * self.value_function.get(next_state_key, 0))
                    for prob, next_state_key, reward in transitions
                )
                delta = max(delta, abs(v - self.value_function[state_key]))
            if delta < self.theta:
                break

    def policy_improvement(self, environment):
        '''Improve the policy based on the current value function.'''
        policy_stable = True
        for state_key in self.policy.keys():
            old_action = self.policy[state_key]
            action_values = []
            for action in range(len(ACTIONS)):
                transitions = environment.get_transitions(state_key, action)
                action_value = sum(
                    prob * (reward + self.gamma * self.value_function.get(next_state_key, 0))
                    for prob, next_state_key, reward in transitions
                )
                action_values.append(action_value)
            best_action = np.argmax(action_values)
            self.policy[state_key] = best_action
            if old_action != best_action:
                policy_stable = False
        return policy_stable

    def train(self, environment, full_game_episodes=10):
        '''Train the agent by playing full games only.'''

        print("Starting full game training...")
        for episode in range(full_game_episodes):
            environment.game.reset()
            done = False
            state = environment.hash_state(environment.game.matris.get_state_features())
            total_reward = 0

            while not done:
                try:
                    action = self.act(state, environment)  # Pass the environment here
                    next_state, reward, done, _ = environment.game.step(action)
                    next_state_key = environment.hash_state(next_state)

                    # Trigger a matrix update after every step
                    environment.game.matris.update(0)
                    pygame.display.flip()

                    # Update value function using observed reward and next state
                    self.value_function[state] = reward + self.gamma * self.value_function.get(next_state_key, 0)

                    state = next_state_key  # Move to the next state
                    total_reward += reward  # Track the total reward for the episode
                except GameOver:
                    done = True  # End training for this episode

            print(f"Episode {episode + 1}/{full_game_episodes} completed. Total Reward: {total_reward}")

        print("Training completed.")






    def act(self, state_key, environment):
        '''Explore all possible placements for the current tetromino and select the best one.'''
        best_action_sequence = []
        max_reward = float('-inf')

        # Access the game through the environment
        game = environment.game

        # Simulate all possible placements
        for rotation in range(4):  # Explore all 4 possible rotations
            for horizontal_shift in range(-MATRIX_WIDTH, MATRIX_WIDTH): 
                shift = horizontal_shift if horizontal_shift % 2 == 0 else -horizontal_shift # Explore left and right shifts
                # Copy the game state to simulate actions
                simulated_game = environment.game.matris.copy()  # Add a method to copy the game state
                try:
                    # Rotate the tetromino
                    for _ in range(rotation):
                        simulated_game.request_rotation()

                    # Move the tetromino horizontally
                    for _ in range(abs(horizontal_shift)):
                        if horizontal_shift < 0:
                            simulated_game.request_movement('left')
                        elif horizontal_shift > 0:
                            simulated_game.request_movement('right')

                    # Drop the tetromino
                    simulated_game.hard_drop()

                    # Evaluate the resulting state
                    new_state = simulated_game.get_state_features()
                    old_state = {'lines': game.matris.lines, 'score': game.matris.score}
                    reward = game.matris.compute_reward(old_state, {'lines_cleared': simulated_game.lines_cleared_last})

                    if reward > max_reward:
                        max_reward = reward
                        best_action_sequence = [
                            (pygame.K_UP, rotation),
                            (pygame.K_LEFT if horizontal_shift < 0 else pygame.K_RIGHT, abs(horizontal_shift)),
                            (pygame.K_SPACE, 1),
                        ]
                except GameOver:
                    pass  # Skip invalid placements

        # Perform the best sequence of actions step-by-step
        for action, count in best_action_sequence:
            for _ in range(count):
                try:
                    environment.game.step(action)  # Apply each action immediately
                    game.matris.update(0)  # Trigger a matrix update after each action
                    pygame.display.flip()  # Refresh the display
                except GameOver:
                    return pygame.K_SPACE  # Return hard drop if the game ends

        return pygame.K_SPACE  # Always return hard drop as the final action






class MatrisEnvironment:
    def __init__(self, game):
        self.game = game

    def hash_state(self, state):
        '''Convert a state (feature vector) into a hashable key.'''
        return tuple(state)

    def unhash_state(self, state_key):
        '''Convert a hashed state key back into a feature vector.'''
        return np.array(state_key)

    def get_transitions(self, state_key, action):
        '''Simulate transitions for a given state-action pair.'''
        # Simulate the action in the current game state
        try:
            next_state, reward, done, _ = self.game.step(ACTIONS[action])
            next_state_key = self.hash_state(next_state)
            return [(1.0, next_state_key, reward)]
        except GameOver:
            # Handle game over with a large penalty
            return [(1.0, None, -1000)]

    def sample_states(self, num_states=100):
        '''Generate a set of states by playing random games.'''
        sampled_states = set()

        for _ in range(num_states):  # Play multiple episodes to sample states
            self.game.reset()
            done = False

            while not done:
                action = random.choice(ACTIONS)  # Perform random actions
                try:
                    state, _, done, _ = self.game.step(action)
                    sampled_states.add(self.hash_state(state))
                except GameOver:
                    done = True

        return list(sampled_states)



def evaluate_policy(agent, game, episodes=10):
    '''Evaluate the agent's policy by playing episodes.'''
    for episode in range(episodes):
        game.reset()
        state = game.matris.get_state_features()
        done = False
        total_reward = 0

        while not done:
            state_key = environment.hash_state(state)
            action = agent.act(state_key, environment)
            try:
                state, reward, done, _ = game.step(action)
                total_reward += reward
            except GameOver:
                done = True

        print(f"Episode {episode + 1}/{episodes}: Total Reward: {reward}")


if __name__ == "__main__":
    pygame.init()
    game = Game(render=True)  # Use render=True to visualize
    environment = MatrisEnvironment(game)
    agent = PolicyIterationAgent()

    # Train the agent
    agent.train(environment, full_game_episodes=100)

    # Evaluate the trained policy
    evaluate_policy(agent, game, environment, episodes=10)


