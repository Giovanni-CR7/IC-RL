import os
import csv
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from train_with_goal_delta import make_env, resolve_model_path


def evaluate(model_path: str, size: int, obstacles: int, max_steps: int,
             num_episodes: int = 100, deterministic: bool = False):
    env = make_env(size, obstacles, max_steps)
    model = PPO.load(model_path)

    results = []
    for i in range(num_episodes):
        obs, info = env.reset()
        done = truncated = False
        steps = 0
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, done, truncated, info = env.step(int(action))
            steps += 1
        results.append((i + 1, info["coverage"], steps))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model name or path")
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--obstacles", type=int, default=12)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--out", default="results/eval_10x10.csv")
    args = parser.parse_args()

    model_path = resolve_model_path(args.model)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Run stochastic and deterministic evaluations
    rows = []
    for det in (False, True):
        res = evaluate(model_path, args.size, args.obstacles, args.max_steps, args.episodes, deterministic=det)
        for ep, cov, steps in res:
            rows.append({
                "deterministic": int(det),
                "episode": ep,
                "coverage": cov,
                "steps": steps,
            })

    # write CSV
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["deterministic", "episode", "coverage", "steps"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # summary
    coverages = [r["coverage"] for r in rows]
    print(f"Saved {len(rows)} evaluations to {args.out}")
    print(f"Mean coverage: {np.mean(coverages)*100:.2f}% ± {np.std(coverages)*100:.2f}%")


if __name__ == "__main__":
    main()
