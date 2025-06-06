import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import DinoNet
from src.constants import *


def expert_policy(state):

    distance, height, kind, y_pos, dino_y, dino_state = state
    # Jump earlier for trees
    if kind == 0 and distance < 100 and dino_y >= (GROUND_Y - DINO_HEIGHT):
        return 1  # jump
    # Duck earlier for birds
    if kind == 1 and distance < 100 and y_pos > 140:
        return 2  # duck

    if dino_state == 2:
        return 2

    return 0  # do nothing


def generate_dataset(env):
    states = []
    actions = []

    for episode in range(100):  # Collect more for better learning
        obs = env.reset()
        done = False
        while not done:
            # if len(states) < 10:  # Only print for the first few for sanity check
            #     print("obs:", obs)
            action = expert_policy(obs)
            states.append(obs)
            actions.append(action)
            obs, reward, done, info = env.step(action)

    x = np.array(states, dtype=np.float32)
    y = np.array(actions, dtype=np.int64)

    unique, counts = np.unique(actions, return_counts=True)
    print("Action distribution:", dict(zip(unique, counts)))

    return x, y


def train(states, actions):
    dataset = TensorDataset(torch.tensor(states), torch.tensor(actions))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    net = DinoNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(10):  # Increase epochs if needed
        for xb, yb in loader:
            logits = net(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}: loss = {loss.item():.4f}")

    return net
