import argparse
import torch
from src.model import DinoNet
from src.dino_env import DinoEnv


def evaluate(model_path, render=True, episodes=5):
    net = DinoNet()
    if model_path == 'checkpoint.pt':
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['model_state_dict'])
    else:
        net.load_state_dict(torch.load(model_path))
    net.eval()

    scores = []
    for ep in range(episodes):
        env = DinoEnv(render_mode=render)
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                q_vals = net(obs_tensor)
                action = q_vals.argmax(dim=1).item()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if render:
                env.render()
        env.close()
        scores.append(total_reward)
        print(f"Episode {ep+1}: Score = {total_reward}")

    avg_score = sum(scores) / len(scores)
    print(f"\nAverage score over {episodes} episodes: {avg_score:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['checkpoint', 'best'], default='best',
                        help='Which model to evaluate: "checkpoint" for last checkpoint, "best" for best_model.pt')
    parser.add_argument('--episodes', type=int, default=5, help='Number of evaluation episodes')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering (run headless)')
    args = parser.parse_args()

    model_file = 'checkpoint.pt' if args.model == 'checkpoint' else 'best_model.pt'
    evaluate(model_file, render=not args.no_render, episodes=args.episodes)
