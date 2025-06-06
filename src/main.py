import torch
from dino_env import DinoEnv
from model import DinoNet
import matplotlib.pyplot as plt
from nn_visualizer import NeuralNetworkVisualizer

if __name__ == "__main__":
    env = DinoEnv(render_mode=True)
    env.seed(42)
    net = DinoNet()
    viz = NeuralNetworkVisualizer(input_dim=6, hidden_dim=16, output_dim=3)

    obs = env.reset()
    done = False

    plt.ion()  # Turn on interactive mode

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = net(obs_tensor)
        action = logits.argmax(dim=1).item()
        obs, reward, done, info = env.step(action)

        # Extract activations for visualization
        input_vals = obs_tensor[0].numpy()
        hidden_vals = net.activations['fc1'][0]
        output_vals = logits[0].numpy()
        env.render()
        viz.draw(input_vals, hidden_vals, output_vals)
        plt.pause(0.001)  # <- Add this line

    plt.ioff()
    plt.show()
    env.close()
