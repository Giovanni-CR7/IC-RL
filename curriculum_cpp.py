#
# curriculum_cpp.py  —  Curriculum Learning para CPP
#
# Treina progressivamente do grid menor para o maior.
# Cada estágio herda os pesos do estágio anterior.
#
# Uso:
#   python curriculum_cpp.py               → roda o curriculum completo do zero
#   python curriculum_cpp.py <model.zip>   → retoma a partir de um modelo existente
#

import sys
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from datetime import datetime

from gymnasium_env.grid_world_cpp_v2 import GridWorldCPPEnvV2

try:
    gym.register(id="gymnasium_env/GridWorldCPPv2-v0", entry_point=GridWorldCPPEnvV2)
except Exception:
    pass

# ── Definição do curriculum ───────────────────────────────────────────────────
#
# Cada estágio: (size, obstacles, max_steps, total_timesteps)
#
# Densidade de obstáculos ~12% em todos os estágios.
# max_steps = 2 * size² (tempo proporcional à área).
# Timesteps maiores nos estágios finais (problema mais complexo).

CURRICULUM = [
    #  size  obs  max_steps  timesteps
    (   5,    3,     50,       300_000 ),
    (   8,    8,    128,       500_000 ),
    (  10,   12,    200,     1_000_000 ),
    (  15,   27,    450,     2_000_000 ),
    (  20,   48,    800,     4_000_000 ),
]

ENTROPY_COEF = 0.05
LOCAL_VIEW   = 5

# ── helpers ───────────────────────────────────────────────────────────────────

def make_env(size, obstacles, max_steps):
    return gym.make(
        "gymnasium_env/GridWorldCPPv2-v0",
        size=size,
        obs_quantity=obstacles,
        max_steps=max_steps,
        local_view_size=LOCAL_VIEW,
        render_mode="rgb_array",
    )

def stage_name(size, obstacles, max_steps):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"ppo_cpp_curriculum_{size}x{size}_obs{obstacles}_steps{max_steps}_{ts}"

# ── ponto de entrada ──────────────────────────────────────────────────────────

resume_path = sys.argv[1] if len(sys.argv) == 2 else None

model = None
start_stage = 0

if resume_path:
    # Descobre em qual estágio retomar pelo nome do arquivo
    for i, (size, obs, max_steps, _) in enumerate(CURRICULUM):
        tag = f"_{size}x{size}_"
        if tag in resume_path:
            start_stage = i + 1
            break
    print(f"Retomando a partir do estágio {start_stage + 1} ({resume_path})")

for i, (size, obs, max_steps, timesteps) in enumerate(CURRICULUM):
    if i < start_stage:
        continue

    print(f"\n{'='*60}")
    print(f"  Estágio {i+1}/{len(CURRICULUM)} — grid {size}×{size}  "
          f"obstáculos: {obs}  max_steps: {max_steps}  timesteps: {timesteps:,}")
    print(f"{'='*60}")

    env  = make_env(size, obs, max_steps)
    name = stage_name(size, obs, max_steps)

    if model is None and resume_path:
        # Primeiro estágio após retomada: carrega modelo salvo
        model = PPO.load(resume_path, env=env, device="cpu")
        print(f"  Pesos carregados de {resume_path}")
    elif model is None:
        # Primeiro estágio do zero
        check_env(env)
        model = PPO("MultiInputPolicy", env, verbose=1,
                    ent_coef=ENTROPY_COEF, device="cpu")
        print("  Novo modelo criado.")
    else:
        # Estágios seguintes: transfere pesos para o novo ambiente
        model.set_env(env)
        print(f"  Pesos transferidos do estágio anterior.")

    log_dir = f"log/{name}"
    model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))

    model.learn(total_timesteps=timesteps, reset_num_timesteps=(i == 0 and not resume_path))

    save_path = f"data/{name}.zip"
    model.save(save_path)
    print(f"\n  ✓ Modelo salvo: {save_path}")
    print(f"  ✓ Logs:         {log_dir}")

print(f"\n{'='*60}")
print("  Curriculum completo!")
print(f"{'='*60}")