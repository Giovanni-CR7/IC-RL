#!/usr/bin/env python3
"""
compare.py — Comparador RL vs. algoritmos clássicos de CPP

Uso
---
  # comparar um modelo contra TODOS os algoritmos:
  python compare.py --model data/meu_modelo.zip --dim 10 --obstacles 8 --max-steps 200 --episodes 100

  # comparar contra um algoritmo específico:
  python compare.py --model data/meu_modelo.zip --dim 10 --obstacles 8 --max-steps 200 --algo frontier

  # só os clássicos, sem RL (ex.: se ainda não treinou aquele grid):
  python compare.py --no-model --dim 20 --obstacles 32 --max-steps 800 --episodes 50

  # salvar CSV
  python compare.py --model data/m.zip --dim 10 --obstacles 8 --max-steps 200 --csv resultados_10x10.csv

"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from baselines.common import make_env, execute_actions, summarize
from baselines import bfs_cover, boustrophedon, spanning_tree, frontier

ALGOS = {
    "bfs": ("BFS (caminho mínimo até não-visitada mais próxima)", bfs_cover.plan),
    "boustrophedon": ("Boustrophedon (varredura zigue-zague)", boustrophedon.plan),
    "spanning_tree": ("Spanning Tree Coverage (travessia de árvore)", spanning_tree.plan),
    "frontier": ("Frontier-based (expansão pela fronteira mais próxima)", frontier.plan),
}


# ── RL ──────────────────────────────────────────────────────────────────────

def _load_rl(path):
    """Carrega PPO ou RecurrentPPO conforme o .zip."""
    from stable_baselines3 import PPO
    try:
        return PPO.load(path), False
    except Exception:
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO.load(path), True


def eval_rl(model, is_rec, env, episodes, seed):
    import numpy as _np
    coverages, steps_list, full = [], [], 0
    for i in range(episodes):
        s = (seed + i) if seed is not None else None
        obs, info = env.reset(seed=s)
        lstm_states = None
        episode_start = _np.ones((1,), dtype=bool)
        done = truncated = False
        steps = 0
        while not done and not truncated:
            if is_rec:
                action, lstm_states = model.predict(
                    obs, state=lstm_states, episode_start=episode_start,
                    deterministic=True)
                episode_start = _np.zeros((1,), dtype=bool)
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, info = env.step(int(action))
            steps += 1
        coverages.append(info["coverage"])
        steps_list.append(steps)
        if done and not truncated:
            full += 1
    return coverages, steps_list, full


# ── clássicos ───────────────────────────────────────────────────────────────

def eval_classic(plan_fn, env, episodes, seed):
    coverages, steps_list, full = [], [], 0
    for i in range(episodes):
        s = (seed + i) if seed is not None else None
        env.reset(seed=s)
        base = env.unwrapped
        actions = plan_fn(base)
        # re-reset com a MESMA semente para executar do estado inicial limpo
        env.reset(seed=s)
        cov, steps, complete = execute_actions(env, actions)
        coverages.append(cov)
        steps_list.append(steps)
        full += int(complete)
    return coverages, steps_list, full


# ── main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Compara RL vs. algoritmos clássicos de CPP.")
    p.add_argument("--model", type=str, default=None, help="Caminho do modelo .zip (RL).")
    p.add_argument("--no-model", action="store_true", help="Não avalia RL, só os clássicos.")
    p.add_argument("--dim", type=int, required=True)
    p.add_argument("--obstacles", type=int, required=True)
    p.add_argument("--max-steps", type=int, required=True)
    p.add_argument("--local-view", type=int, default=5)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0,
                   help="Semente inicial (mesmos mapas para RL e clássicos). Default 0.")
    p.add_argument("--algo", type=str, default="all",
                   choices=list(ALGOS.keys()) + ["all"],
                   help="Algoritmo clássico a comparar (ou 'all').")
    p.add_argument("--csv", type=str, default=None, help="Salva os resultados neste CSV.")
    args = p.parse_args()

    rows = []

    # 1) RL
    if not args.no_model:
        if not args.model:
            p.error("Forneça --model CAMINHO.zip ou use --no-model.")
        model, is_rec = _load_rl(args.model)
        env = make_env(args.dim, args.obstacles, args.max_steps, args.local_view)
        cov, steps, full = eval_rl(model, is_rec, env, args.episodes, args.seed)
        env.close()
        tag = "RL (RecurrentPPO/LSTM)" if is_rec else "RL (PPO-MLP)"
        rows.append(summarize("rl", tag, args.dim, args.obstacles, args.max_steps,
                              args.episodes, cov, steps, full))

    # 2) clássicos
    algo_keys = list(ALGOS.keys()) if args.algo == "all" else [args.algo]
    for key in algo_keys:
        label, plan_fn = ALGOS[key]
        env = make_env(args.dim, args.obstacles, args.max_steps, args.local_view)
        cov, steps, full = eval_classic(plan_fn, env, args.episodes, args.seed)
        env.close()
        rows.append(summarize(key, label, args.dim, args.obstacles, args.max_steps,
                              args.episodes, cov, steps, full))

    # 3) tabela final + CSV
    print(f"\n{'='*66}")
    print(f"  TABELA COMPARATIVA — grid {args.dim}x{args.dim}, {args.episodes} episódios")
    print(f"{'='*66}")
    print(f"  {'Algoritmo':<34} {'Cob.compl.':>10} {'Cob.média':>11} {'Passos':>10}")
    print(f"  {'-'*34} {'-'*10} {'-'*11} {'-'*10}")
    for r in rows:
        print(f"  {r['algo']:<34} {r['full_rate']*100:>9.1f}% "
              f"{r['mean_cov']*100:>9.1f}% {r['mean_steps']:>9.1f}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["algoritmo", "grid", "obstaculos", "max_steps", "episodios",
                        "cobertura_completa_pct", "cobertura_media_pct",
                        "cobertura_std_pct", "passos_medios", "passos_std"])
            for r in rows:
                w.writerow([r["algo"], f"{args.dim}x{args.dim}", args.obstacles,
                            args.max_steps, args.episodes,
                            f"{r['full_rate']*100:.1f}", f"{r['mean_cov']*100:.2f}",
                            f"{r['std_cov']*100:.2f}", f"{r['mean_steps']:.1f}",
                            f"{r['std_steps']:.1f}"])
        print(f"\n  CSV salvo em {args.csv}")


if __name__ == "__main__":
    main()
