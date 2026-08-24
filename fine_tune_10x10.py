import os
import sys
from datetime import datetime

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

from train_cpp import make_env, resolve_model_path


def fine_tune(model_name: str, timesteps: int = 500_000,
              size: int = 10, obstacles: int = 12, max_steps: int = 200):
    model_path = resolve_model_path(model_name)
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path, device="cpu")

    env = make_env(size, obstacles, max_steps)
    model.set_env(env)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"{os.path.splitext(os.path.basename(model_path))[0]}_finetuned_{timesteps}_{ts}.zip"
    out_path = os.path.join("data", out_name)
    log_dir = os.path.join("log", os.path.splitext(out_name)[0])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))
    print(f"Starting fine-tune: {timesteps} timesteps on {size}x{size}")
    model.learn(total_timesteps=timesteps, reset_num_timesteps=False)
    model.save(out_path)
    print(f"Fine-tuned model saved to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fine_tune_10x10.py <model_name_or_path> [timesteps]")
        sys.exit(1)
    model = sys.argv[1]
    ts = int(sys.argv[2]) if len(sys.argv) >= 3 else 500_000
    fine_tune(model, ts)
