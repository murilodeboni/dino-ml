import torch

from src.dino_env import DinoEnv
from src.train_dqn import train_dqn

if __name__ == "__main__":
    import argparse
    import src.supervised as sup

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['supervised', 'dqn'], default='dqn')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--fresh', action='store_true', help='Start training from scratch (ignore checkpoint)')
    args = parser.parse_args()

    if args.mode == 'supervised':
        env = DinoEnv(render_mode=False)
        states, actions = sup.generate_dataset(env)
        net = sup.train(states, actions)
        # Test visually
        env = DinoEnv(render_mode=args.render)
        env.seed(42)
        sup.test_model(env, net, render=args.render)
    elif args.mode == 'dqn':
        net = train_dqn(num_episodes=1001, render=args.render, fresh=args.fresh)
        # Test the trained DQN agent
        env = DinoEnv(render_mode=True)
        obs = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                q_vals = net(obs_tensor)
                action = q_vals.argmax(dim=1).item()
            obs, reward, done, info = env.step(action)
            env.render()
        env.close()
