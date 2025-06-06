import torch
from dino_env import DinoEnv
import matplotlib.pyplot as plt
from nn_visualizer import NeuralNetworkVisualizer
import src.supervised as sup

SHOW_GAME = True           # Toggle game window on/off
SHOW_NEURONS = False        # Toggle neural network visualization

if __name__ == "__main__":
    # Dataset creation and training (no rendering needed here)
    env = DinoEnv(render_mode=False)
    states, actions = sup.generate_dataset(env)
    net = sup.train(states, actions)

    # Now, test with desired rendering and visualization
    env = DinoEnv(render_mode=SHOW_GAME)
    env.seed(42)
    viz = NeuralNetworkVisualizer(input_dim=6, hidden_dim=16, output_dim=3) if SHOW_NEURONS else None

    obs = env.reset()
    done = False

    if SHOW_NEURONS:
        plt.ion()  # Interactive plotting on

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = net(obs_tensor)
        action = logits.argmax(dim=1).item()
        obs, reward, done, info = env.step(action)

        if SHOW_GAME:
            env.render()
        if SHOW_NEURONS:
            input_vals = obs_tensor[0].numpy()
            hidden_vals = net.activations['fc1'][0]
            output_vals = logits[0].numpy()
            viz.draw(input_vals, hidden_vals, output_vals)
            # plt.pause(0.00001)  # For live updating

    if SHOW_NEURONS:
        plt.ioff()
        plt.show()
    env.close()
