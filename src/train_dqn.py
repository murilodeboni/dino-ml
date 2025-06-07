# train_dqn.py

import numpy as np
import torch
import random
from collections import deque
from src.model import DinoNet
from src.dino_env import DinoEnv


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buffer)


def train_dqn(num_episodes=500, max_steps=1000, render=False):
    env = DinoEnv(render_mode=render)
    net = DinoNet()
    target_net = DinoNet()
    target_net.load_state_dict(net.state_dict())
    target_net.eval()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    buffer = ReplayBuffer(capacity=5000)
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    target_update_freq = 10

    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        for step in range(max_steps):
            if render:
                env.render()
            # Epsilon-greedy action
            if random.random() < epsilon:
                action = random.choice([0, 1, 2])
            else:
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    q_vals = net(obs_tensor)
                    action = q_vals.argmax(dim=1).item()
            next_obs, reward, done, _ = env.step(action)
            buffer.push((obs, action, reward, next_obs, float(done)))
            obs = next_obs
            total_reward += reward

            # DQN update
            if len(buffer) > batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.int64)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32)

                q_values = net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = target_net(next_states).max(1)[0]
                targets = rewards + gamma * next_q_values * (1 - dones)
                loss = torch.nn.functional.mse_loss(q_values, targets.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        # Epsilon decay
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Target network update
        if episode % target_update_freq == 0:
            target_net.load_state_dict(net.state_dict())

        print(f"Episode {episode + 1}: reward = {total_reward}")

    env.close()
    return net


