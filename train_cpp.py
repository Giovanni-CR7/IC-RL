#!/usr/bin/env python3
"""
train_cpp.py — Treino/teste/execução PPO para Coverage Path Planning (CPP)

Uso
---
  # PPO-MLP (padrão, sem memória):
  python train_cpp.py train --dim 10 --obstacles 12 --max-steps 200 --timesteps 500000
  # RecurrentPPO (PPO+LSTM, com memória):
  python train_cpp.py train --dim 10 --obstacles 12 --max-steps 200 --timesteps 500000 --recurrent
  python train_cpp.py test  --dim 10 --obstacles 12 --max-steps 200 --model NOME_OU_CAMINHO
  python train_cpp.py run   --dim 10 --obstacles 12 --max-steps 200 --model NOME_OU_CAMINHO
  python train_cpp.py curriculum [--recurrent] [--resume data/modelo.zip]

modelos são salvos em `data/`, logs em `log/`.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback

try:
    from sb3_contrib import RecurrentPPO
    _HAS_RECURRENT = True
except ImportError:
    RecurrentPPO = None
    _HAS_RECURRENT = False

from gymnasium_env.grid_world_cpp_v2 import GridWorldCPPEnvV2

ENV_ID = "gymnasium_env/GridWorldCPPv2-v0"




class RecurrentState:
    """Guarda o estado oculto da LSTM entre chamadas de predict dentro de um
    episódio. Para o modelo MLP, é um no-op inofensivo (state permanece None).

    Uso:
        rs = RecurrentState()            # no início de cada episódio
        action = policy_predict(model, obs, rs, deterministic=True)
    """
    __slots__ = ("lstm_states", "episode_start")

    def __init__(self):
        self.lstm_states = None
        # True no primeiro passo do episódio → sinaliza à LSTM para resetar
        self.episode_start = np.ones((1,), dtype=bool)


def is_recurrent(model) -> bool:
    return _HAS_RECURRENT and isinstance(model, RecurrentPPO)


def policy_predict(model, obs, rstate: "RecurrentState", deterministic: bool = True):
    """Prediz uma ação lidando com o estado da LSTM de forma transparente.

    - RecurrentPPO: passa/recebe lstm_states e episode_start (obrigatório para
      a memória funcionar; sem isso a LSTM "esquece" a cada passo).
    - PPO comum: ignora o estado recorrente (o predict do PPO aceita só obs).

    Retorna apenas a ação; o estado da LSTM é atualizado dentro de `rstate`.
    """
    if is_recurrent(model):
        action, rstate.lstm_states = model.predict(
            obs,
            state=rstate.lstm_states,
            episode_start=rstate.episode_start,
            deterministic=deterministic,
        )
        # depois do primeiro passo, não é mais início de episódio
        rstate.episode_start = np.zeros((1,), dtype=bool)
    else:
        action, _ = model.predict(obs, deterministic=deterministic)
    return action


def load_model(path: str):
    """Carrega um .zip tentando primeiro PPO; se falhar, tenta RecurrentPPO.
    Assim test/run funcionam sem o usuário precisar repetir a flag --recurrent
    ao avaliar um modelo com LSTM."""
    try:
        return PPO.load(path)
    except Exception as e_ppo:
        if _HAS_RECURRENT:
            try:
                return RecurrentPPO.load(path)
            except Exception as e_rec:
                raise RuntimeError(
                    f"Falha ao carregar como PPO ({e_ppo}) e como "
                    f"RecurrentPPO ({e_rec})."
                )
        raise


# ── ambiente ──────────────────────────────────────────────────────────────

def register_env():
    try:
        gym.register(id=ENV_ID, entry_point=GridWorldCPPEnvV2)
    except Exception:
        pass  


def make_env(size, obstacles, max_steps, local_view=5, render_mode="rgb_array"):
    register_env()
    return gym.make(
        ENV_ID,
        size=size,
        obs_quantity=obstacles,
        max_steps=max_steps,
        local_view_size=local_view,
        render_mode=render_mode,
    )


def resolve_model_path(name: str) -> str:
    """Resolve um nome/caminho de modelo para um arquivo existente.
    Tenta, em ordem: nome como dado, nome+'.zip', data/nome, data/nome+'.zip'.
    """
    candidates = []
    if os.path.isabs(name) or name.startswith("data" + os.sep) or name.startswith("data/"):
        candidates.append(name)
        if not name.endswith(".zip"):
            candidates.append(name + ".zip")
    else:
        candidates.append(name)
        candidates.append(name + ".zip")
        candidates.append(os.path.join("data", name))
        candidates.append(os.path.join("data", name + ".zip"))

    for p in candidates:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(f"Modelo não encontrado. Tentativas: {candidates}")


# ── trava anti-deadlock (fallback em tempo de inferência) ──────────────────
#
# Uma política determinística sem memória pode ficar presa repetindo
# indefinidamente a mesma ação inválida (ex.: colidir com um obstáculo) num
# "beco sem saída" nunca visto em treino — o estado observado praticamente
# não muda de um passo para o outro, então o argmax da rede não muda.
# como a observação já carrega a direção-objetivo obstacle-aware (calculada
# via BFS dentro do ambiente, ver grid_world_cpp_v2.py), essa trava não
# precisa recalcular nada: se o agente passar `stuck_patience` passos sem
# cobrir nenhuma célula nova, ela simplesmente usa esse sinal já disponível
# em vez da ação (travada) do modelo, só para aquele passo.
_DIRECTION_TO_ACTION = {
    (1, 0): 0,   # direita
    (0, -1): 1,  # cima
    (-1, 0): 2,  # esquerda
    (0, 1): 3,   # baixo
}


def direction_to_action(dx: float, dy: float, fallback: int) -> int:
    key = (int(round(dx)), int(round(dy)))
    return _DIRECTION_TO_ACTION.get(key, fallback)


# ── hiperparâmetros adaptativos ──────────────────────────────────────────

def compute_gamma(max_steps: int) -> float:
    """Gamma adaptativo: episódios mais longos exigem horizonte de desconto
    maior para o agente 'enxergar' recompensas distantes no tempo (ex.:
    cobrir uma célula que só compensa vários passos depois)."""
    g = 1.0 - 1.0 / (2.0 * max_steps)
    return float(np.clip(g, 0.99, 0.999))


def compute_n_steps(max_steps: int) -> int:
    """n_steps do rollout do PPO: tenta cobrir pelo menos ~2 episódios
    completos por rollout, dentro de limites razoáveis de memória/velocidade."""
    n = max_steps * 2
    n = max(1024, min(4096, n))
    return int(round(n / 64) * 64)  # múltiplo de 64 para facilitar minibatches


def create_ppo_model(env, max_steps: int, ent_coef_start: float = 0.02,
                     recurrent: bool = False, lstm_hidden: int = 256, verbose: int = 1):
    """Cria o modelo. recurrent=False → PPO-MLP (sem memória, baseline).
    recurrent=True → RecurrentPPO (PPO+LSTM), que carrega memória do que já
    foi observado dentro do episódio.

    Observação: com LSTM, batch_size deve ser múltiplo de n_steps (o buffer
    recorrente é organizado por sequências), então usamos batch_size=n_steps
    para o modo recorrente em vez do 64 fixo do MLP."""
    gamma = compute_gamma(max_steps)
    n_steps = compute_n_steps(max_steps)

    if recurrent:
        if not _HAS_RECURRENT:
            raise ImportError(
                "RecurrentPPO requer o pacote sb3-contrib. Instale com:\n"
                "    pip install sb3-contrib"
            )
        model = RecurrentPPO(
            "MultiInputLstmPolicy",   # suporta observação Dict (local + global)
            env,
            verbose=verbose,
            gamma=gamma,
            n_steps=n_steps,
            batch_size=n_steps,        # múltiplo de n_steps p/ buffer recorrente
            ent_coef=ent_coef_start,
            policy_kwargs=dict(
                net_arch=dict(pi=[128, 128], vf=[128, 128]),
                lstm_hidden_size=lstm_hidden,
                n_lstm_layers=1,
                enable_critic_lstm=True,   # LSTM separada para o crítico
            ),
            device="cpu",
        )
        print(f"  RecurrentPPO (LSTM) criado — gamma={gamma:.4f}  n_steps={n_steps}  "
              f"lstm_hidden={lstm_hidden}  ent_coef inicial={ent_coef_start}")
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=verbose,
            gamma=gamma,
            n_steps=n_steps,
            batch_size=64,
            ent_coef=ent_coef_start,
            policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
            device="cpu",
        )
        print(f"  PPO-MLP criado — gamma={gamma:.4f}  n_steps={n_steps}  ent_coef inicial={ent_coef_start}")
    return model


# ── callbacks ─────────────────────────────────────────────────────────────

class EntCoefScheduler(BaseCallback):
    """Decai linearmente o coeficiente de entropia do PPO ao longo do
    treino: mais exploração no início, mais exploração de política
    quase-determinística no fim — resolve o problema de ent_coef fixo alto
    impedir a convergência."""

    def __init__(self, start: float = 0.02, end: float = 0.001, verbose: int = 0):
        super().__init__(verbose)
        self.start = start
        self.end = end
        self._total = None

    def _on_training_start(self) -> None:
        self._total = self.locals.get("total_timesteps", 1) or 1

    def _on_step(self) -> bool:
        frac = min(1.0, self.num_timesteps / self._total)
        self.model.ent_coef = self.start + frac * (self.end - self.start)
        return True


class CoverageMonitorCallback(BaseCallback):
    """A cada `eval_freq` passos, roda `n_episodes` episódios determinísticos
    numa cópia limpa do ambiente e loga cobertura média / taxa de cobertura
    completa — sinal direto de qualidade, visível no TensorBoard/CSV
    (métricas 'eval/mean_coverage' e 'eval/full_coverage_rate'), em vez de
    depender só do reward médio para saber se o agente está indo bem."""

    def __init__(self, eval_env_fn, n_episodes: int = 5, eval_freq: int = 20_000, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env_fn = eval_env_fn
        self.n_episodes = n_episodes
        self.eval_freq = eval_freq
        self._last_eval = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval < self.eval_freq:
            return True
        self._last_eval = self.num_timesteps

        env = self.eval_env_fn()
        coverages = []
        full = 0
        for _ in range(self.n_episodes):
            obs, info = env.reset()
            rstate = RecurrentState()  # reseta memória da LSTM por episódio
            done = truncated = False
            while not done and not truncated:
                action = policy_predict(self.model, obs, rstate, deterministic=True)
                obs, _, done, truncated, info = env.step(int(action))
            coverages.append(info["coverage"])
            if done and not truncated:
                full += 1
        env.close()

        mean_cov = float(np.mean(coverages))
        full_rate = full / self.n_episodes
        self.logger.record("eval/mean_coverage", mean_cov)
        self.logger.record("eval/full_coverage_rate", full_rate)
        if self.verbose:
            print(f"  [eval @ {self.num_timesteps}] cobertura média: {mean_cov:.1%} "
                  f"| cobertura completa: {full_rate:.1%}")
        return True


# ── comandos ──────────────────────────────────────────────────────────────

def make_run_paths(tag, dim, obstacles, max_steps, arch="mlp"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    model_path = f"data/ppo_cpp_{arch}_{dim}_{obstacles}_{max_steps}_{ts}{suffix}.zip"
    log_dir = f"log/ppo_cpp_{arch}_{dim}_{obstacles}_{max_steps}_{ts}{suffix}"
    return model_path, log_dir


def cmd_train(args):
    os.makedirs("data", exist_ok=True)
    os.makedirs("log", exist_ok=True)

    env = make_env(args.dim, args.obstacles, args.max_steps, args.local_view)
    check_env(env)

    model = create_ppo_model(env, args.max_steps, args.ent_coef_start,
                             recurrent=args.recurrent, lstm_hidden=args.lstm_hidden)
    arch = "lstm" if args.recurrent else "mlp"
    model_path, log_dir = make_run_paths("train", args.dim, args.obstacles, args.max_steps, arch)
    model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))

    callbacks = [EntCoefScheduler(args.ent_coef_start, args.ent_coef_end)]
    if not args.no_monitor:
        eval_fn = lambda: make_env(args.dim, args.obstacles, args.max_steps, args.local_view)
        callbacks.append(CoverageMonitorCallback(eval_fn, args.eval_episodes, args.eval_freq, verbose=1))

    print(f"Treinando {args.timesteps:,} passos em grid {args.dim}x{args.dim} "
          f"(obstáculos={args.obstacles}, max_steps={args.max_steps}, "
          f"visão local={args.local_view}x{args.local_view})")
    model.learn(total_timesteps=args.timesteps, callback=callbacks)
    model.save(model_path)
    print(f"Modelo salvo em {model_path}")


def cmd_test(args):
    env = make_env(args.dim, args.obstacles, args.max_steps, args.local_view)
    model = load_model(resolve_model_path(args.model))
    if is_recurrent(model):
        print("Modelo recorrente (LSTM) detectado — estado oculto será mantido "
              "entre passos e resetado a cada episódio.")

    coverages, steps_list, full_count = [], [], 0
    watchdog_triggers_total = 0
    for i in range(args.episodes):
        obs, info = env.reset()
        rstate = RecurrentState()  # reseta memória da LSTM no início do episódio
        done = truncated = False
        steps = 0
        stall = 0
        last_visited = info["visited_cells"]
        while not done and not truncated:
            action = policy_predict(model, obs, rstate, deterministic=True)
            if not args.no_watchdog and stall >= args.stuck_patience:
                dx, dy = obs["global"][4], obs["global"][5]
                action = direction_to_action(dx, dy, fallback=int(action))
                watchdog_triggers_total += 1
            obs, _, done, truncated, info = env.step(int(action))
            steps += 1
            if info["visited_cells"] > last_visited:
                stall = 0
            else:
                stall += 1
            last_visited = info["visited_cells"]
        coverages.append(info["coverage"])
        steps_list.append(steps)
        if done and not truncated:
            full_count += 1
            print(f"Ep {i+1:3d}: cobertura total em {steps} passos")
        else:
            print(f"Ep {i+1:3d}: {info['coverage']:.1%} em {steps} passos")

    print(f"\nResumo do teste ({args.episodes} episódios):")
    print(f"Taxa de cobertura completa: {full_count/args.episodes*100:.1f}% ({full_count}/{args.episodes})")
    print(f"Cobertura média: {np.mean(coverages)*100:.2f}% ± {np.std(coverages)*100:.2f}%")
    print(f"Passos médios: {np.mean(steps_list):.1f} ± {np.std(steps_list):.1f}")
    if not args.no_watchdog:
        print(f"Trava anti-deadlock acionada {watchdog_triggers_total}x no total "
              f"(~{watchdog_triggers_total/args.episodes:.1f}/episódio). "
              f"Quanto maior, mais a cobertura depende do fallback BFS e não da política.")


def cmd_run(args):
    env = make_env(args.dim, args.obstacles, args.max_steps, args.local_view, render_mode="human")
    model = load_model(resolve_model_path(args.model))
    if is_recurrent(model):
        print("Modelo recorrente (LSTM) detectado.")

    obs, info = env.reset()
    rstate = RecurrentState()  # reseta memória da LSTM no início do episódio
    done = truncated = False
    steps = total_reward = 0
    while not done and not truncated:
        action = policy_predict(model, obs, rstate, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
        total_reward += reward
        steps += 1
        print(f"Passo {steps:4d} | reward: {reward:+.2f} | cobertura: {info['coverage']:.1%} | done: {done}")
    print(f"Fim — reward total: {total_reward:.2f} | cobertura: {info['coverage']:.1%}")


CURRICULUM = [
    # size, obstacles, max_steps, timesteps
    (5,  max(1, round(0.12 * 5 * 5)),   50,  200_000),
    (7,  max(1, round(0.12 * 7 * 7)),   98,  300_000),
    (10, max(1, round(0.12 * 10 * 10)), 200, 500_000),
    (15, max(1, round(0.12 * 15 * 15)), 450, 1_000_000),
    (20, max(1, round(0.12 * 20 * 20)), 800, 2_000_000),
]


def cmd_curriculum(args):
    os.makedirs("data", exist_ok=True)
    os.makedirs("log", exist_ok=True)

    # gamma/n_steps calculados a partir do ÚLTIMO estágio (o mais exigente) e
    # mantidos fixos por todo o curriculum — evita ter que realocar o
    # rollout_buffer do PPO no meio do treino a cada transição de estágio.
    final_max_steps = CURRICULUM[-1][2]
    gamma = compute_gamma(final_max_steps)

    model = None
    start_stage = 0
    if args.resume:
        for i, (size, obs, max_steps, ts) in enumerate(CURRICULUM):
            if f"_{size}x{size}_" in args.resume:
                start_stage = i + 1
                break
        print(f"Retomando a partir do estágio {start_stage + 1} ({args.resume})")

    for i, (size, obs, max_steps, timesteps) in enumerate(CURRICULUM):
        if i < start_stage:
            continue

        print(f"\n{'='*60}")
        print(f"  Estágio {i+1}/{len(CURRICULUM)} — grid {size}x{size}  "
              f"obstáculos: {obs}  max_steps: {max_steps}  timesteps: {timesteps:,}")
        print(f"{'='*60}")

        env = make_env(size, obs, max_steps, args.local_view)
        ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        arch = "lstm" if args.recurrent else "mlp"
        name = f"ppo_cpp_{arch}_curr_{size}x{size}_obs{obs}_steps{max_steps}_{ts_tag}"

        if model is None and args.resume:
            # carrega com a classe certa (RecurrentPPO se --recurrent, senão PPO)
            loader = RecurrentPPO if (args.recurrent and _HAS_RECURRENT) else PPO
            model = loader.load(args.resume, env=env, device="cpu")
            model.gamma = gamma
            if getattr(model, "rollout_buffer", None) is not None:
                model.rollout_buffer.gamma = gamma
            print(f"  Pesos carregados de {args.resume} (via {loader.__name__})")
        elif model is None:
            check_env(env)
            model = create_ppo_model(env, final_max_steps, args.ent_coef_start,
                                     recurrent=args.recurrent, lstm_hidden=args.lstm_hidden)
            print("  Novo modelo criado.")
        else:
            model.set_env(env)
            print("  Pesos transferidos do estágio anterior.")

        log_dir = f"log/{name}"
        model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))

        # menos exploração a cada estágio novo (a política já sabe o básico)
        ent_start_stage = max(0.001, args.ent_coef_start / (2 ** i))
        callbacks = [EntCoefScheduler(ent_start_stage, args.ent_coef_end)]
        if not args.no_monitor:
            eval_fn = (lambda size=size, obs=obs, max_steps=max_steps:
                       make_env(size, obs, max_steps, args.local_view))
            callbacks.append(CoverageMonitorCallback(eval_fn, args.eval_episodes, args.eval_freq, verbose=1))

        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            reset_num_timesteps=(i == 0 and args.resume is None),
        )

        save_path = f"data/{name}.zip"
        model.save(save_path)
        print(f"\n  ✓ Modelo salvo: {save_path}")
        print(f"  ✓ Logs:         {log_dir}")

    print(f"\n{'='*60}")
    print("  Curriculum completo!")
    print(f"{'='*60}")


# ── CLI ───────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Treino/teste/execução PPO para Coverage Path Planning.")
    sub = p.add_subparsers(dest="mode", required=True)

    p_train = sub.add_parser("train", help="Treina um modelo do zero num único tamanho de grid.")
    p_train.add_argument("--dim", type=int, required=True)
    p_train.add_argument("--obstacles", type=int, required=True)
    p_train.add_argument("--max-steps", type=int, required=True)
    p_train.add_argument("--timesteps", type=int, required=True)
    p_train.add_argument("--local-view", type=int, default=5, help="Tamanho da janela local (ímpar, ex: 5 ou 7).")
    p_train.add_argument("--ent-coef-start", type=float, default=0.02)
    p_train.add_argument("--ent-coef-end", type=float, default=0.001)
    p_train.add_argument("--eval-freq", type=int, default=20_000)
    p_train.add_argument("--eval-episodes", type=int, default=5)
    p_train.add_argument("--no-monitor", action="store_true", help="Desativa o CoverageMonitorCallback.")
    p_train.add_argument("--recurrent", action="store_true",
                         help="Usa RecurrentPPO (PPO+LSTM) em vez de PPO-MLP. Requer sb3-contrib.")
    p_train.add_argument("--lstm-hidden", type=int, default=256,
                         help="Tamanho do estado oculto da LSTM (só com --recurrent).")
    p_train.set_defaults(func=cmd_train)

    p_test = sub.add_parser("test", help="Avalia um modelo salvo em N episódios (ação determinística).")
    p_test.add_argument("--dim", type=int, required=True)
    p_test.add_argument("--obstacles", type=int, required=True)
    p_test.add_argument("--max-steps", type=int, required=True)
    p_test.add_argument("--model", type=str, required=True, help="Nome ou caminho do modelo (.zip).")
    p_test.add_argument("--local-view", type=int, default=5)
    p_test.add_argument("--episodes", type=int, default=100)
    p_test.add_argument("--no-watchdog", action="store_true",
                        help="Desativa a trava anti-deadlock (avalia a política pura).")
    p_test.add_argument("--stuck-patience", type=int, default=8,
                        help="Passos sem cobrir célula nova antes de a trava assumir.")
    p_test.set_defaults(func=cmd_test)

    p_run = sub.add_parser("run", help="Executa um modelo salvo com renderização visual (pygame).")
    p_run.add_argument("--dim", type=int, required=True)
    p_run.add_argument("--obstacles", type=int, required=True)
    p_run.add_argument("--max-steps", type=int, required=True)
    p_run.add_argument("--model", type=str, required=True)
    p_run.add_argument("--local-view", type=int, default=5)
    p_run.set_defaults(func=cmd_run)

    p_curr = sub.add_parser("curriculum", help="Treina progressivamente em grids 5→7→10→15→20.")
    p_curr.add_argument("--resume", type=str, default=None, help="Caminho de um .zip para retomar o curriculum.")
    p_curr.add_argument("--local-view", type=int, default=5)
    p_curr.add_argument("--ent-coef-start", type=float, default=0.02)
    p_curr.add_argument("--ent-coef-end", type=float, default=0.001)
    p_curr.add_argument("--eval-freq", type=int, default=20_000)
    p_curr.add_argument("--eval-episodes", type=int, default=5)
    p_curr.add_argument("--no-monitor", action="store_true")
    p_curr.add_argument("--recurrent", action="store_true",
                        help="Usa RecurrentPPO (PPO+LSTM) em vez de PPO-MLP. Requer sb3-contrib.")
    p_curr.add_argument("--lstm-hidden", type=int, default=256,
                        help="Tamanho do estado oculto da LSTM (só com --recurrent).")
    p_curr.set_defaults(func=cmd_curriculum)

    return p


if __name__ == "__main__":
    parser = build_parser()
    cli_args = parser.parse_args()
    cli_args.func(cli_args)