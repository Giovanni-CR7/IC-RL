#
# train_grid_world_cpp_v2.py
# Uso: python train_grid_world_cpp_v2.py <train|test|run|curriculum> dim obstacles max_steps [total_timesteps]
#

import sys
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from datetime import datetime

from gymnasium_env.grid_world_cpp_v2 import GridWorldCPPEnvV2


def print_action(action: int) -> str:
    return {0: "direita", 1: "cima", 2: "esquerda", 3: "baixo"}.get(action, "?")


# ── validação de args ──────────────────────────────────────────────────────────

if len(sys.argv) < 2 or sys.argv[1] not in ("train", "test", "run", "curriculum"):
    print("Uso: python train_grid_world_cpp_v2.py <train|test|run|curriculum> dim obstacles max_steps [total_timesteps]")
    sys.exit(1)

mode = sys.argv[1]

if mode in ("train", "curriculum") and len(sys.argv) != 6:
    print(f"Uso ({mode}): python train_grid_world_cpp_v2.py {mode} dim obstacles max_steps total_timesteps")
    sys.exit(1)
if mode in ("test", "run") and len(sys.argv) != 5:
    print(f"Uso ({mode}): python train_grid_world_cpp_v2.py {mode} dim obstacles max_steps")
    sys.exit(1)

DIM        = int(sys.argv[2])
OBSTACLES  = int(sys.argv[3])
MAX_STEPS  = int(sys.argv[4])
TOTAL_TS   = int(sys.argv[5]) if mode in ("train", "curriculum") else None

# Hiperparâmetros PPO
ENTROPY_COEF = 0.05
LOCAL_VIEW   = 5   # janela local 5×5 (pode mudar para 7 em grids >= 15)

# ── registro do ambiente ───────────────────────────────────────────────────────

try:
    gym.register(
        id="gymnasium_env/GridWorldCPPv2-v0",
        entry_point=GridWorldCPPEnvV2,
    )
except Exception:
    pass

ENV_KWARGS = dict(
    size=DIM,
    obs_quantity=OBSTACLES,
    max_steps=MAX_STEPS,
    local_view_size=LOCAL_VIEW,
)

# ── helpers ───────────────────────────────────────────────────────────────────

def make_env(render_mode="rgb_array"):
    return gym.make("gymnasium_env/GridWorldCPPv2-v0", render_mode=render_mode, **ENV_KWARGS)


def make_model_path(tag=""):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    return (
        f"data/ppo_cpp_v2_{DIM}_{OBSTACLES}_{MAX_STEPS}_{ENTROPY_COEF}_{ts}{suffix}.zip",
        f"log/ppo_cpp_v2_{DIM}_{OBSTACLES}_{MAX_STEPS}_{ENTROPY_COEF}_{ts}{suffix}",
    )


# ── modos ─────────────────────────────────────────────────────────────────────

if mode == "train":
    print("─── Treinamento CPP v2 ───")
    env = make_env()
    check_env(env)

    model = PPO("MultiInputPolicy", env, verbose=1, ent_coef=ENTROPY_COEF, device="cpu")
    model_path, log_dir = make_model_path()

    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    print(f"Iniciando com {TOTAL_TS} timesteps...")
    model.learn(total_timesteps=TOTAL_TS)
    model.save(model_path)
    print(f"Modelo salvo em {model_path}")
    print(f"Logs em {log_dir}")


elif mode == "curriculum":
    print("─── Curriculum Learning CPP v2 ───")
    model_name = input("Nome do modelo base (sem .zip): ")
    base_path = f"data/{model_name}.zip"

    env = make_env()
    model = PPO.load(base_path, env=env, device="cpu")

    model_path, log_dir = make_model_path("curriculum")
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    print(f"Continuando com {TOTAL_TS} timesteps...")
    model.learn(total_timesteps=TOTAL_TS, reset_num_timesteps=False)
    model.save(model_path)
    print(f"Modelo salvo em {model_path}")


elif mode == "run":
    model_name = input("Nome do modelo (sem .zip): ")
    model = PPO.load(f"data/{model_name}.zip")
    env = make_env(render_mode="human")

    obs, info = env.reset()
    done = truncated = False
    steps = total_reward = 0

    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, truncated, info = env.step(int(action))
        total_reward += reward
        steps += 1
        print(f"Passo {steps:4d} | ação: {print_action(int(action)):8s} | "
              f"reward: {reward:+.2f} | cobertura: {info['coverage']:.1%} | "
              f"fim: {done} | trunc: {truncated}")

    print(f"\n─── Fim ─── reward total: {total_reward:.2f} | cobertura final: {info['coverage']:.1%}")


elif mode == "test":
    model_name = input("Nome do modelo (sem .zip): ")
    model = PPO.load(f"data/{model_name}.zip")
    env = make_env()

    num_episodes = 100
    full_coverage_count = 0
    coverages, steps_list = [], []

    for i in range(num_episodes):
        obs, info = env.reset()
        done = truncated = False
        steps = 0

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=False)
            obs, _, done, truncated, info = env.step(int(action))
            steps += 1

        coverages.append(info["coverage"])
        steps_list.append(steps)

        if done and not truncated:
            full_coverage_count += 1
            print(f"Ep {i+1:3d}: cobertura total em {steps} passos")
        else:
            print(f"Ep {i+1:3d}: {info['coverage']:.1%} em {steps} passos")

    rate = full_coverage_count / num_episodes * 100
    print(f"\n─── Resultado ({num_episodes} episódios) ───")
    print(f"Taxa de cobertura total : {rate:.1f}%  ({full_coverage_count}/{num_episodes})")
    print(f"Cobertura média         : {np.mean(coverages)*100:.2f}% ± {np.std(coverages)*100:.2f}%")
    print(f"Min / Max cobertura     : {np.min(coverages)*100:.2f}% / {np.max(coverages)*100:.2f}%")
    print(f"Passos médios           : {np.mean(steps_list):.1f} ± {np.std(steps_list):.1f}")
    print(f"Min / Max passos        : {int(np.min(steps_list))} / {int(np.max(steps_list))}")