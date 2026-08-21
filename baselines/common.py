"""
common.py — Utilidades compartilhadas pelos algoritmos clássicos de CPP.

Todos os algoritmos de baseline (BFS, Boustrophedon, Spanning Tree, Frontier)
produzem uma LISTA DE AÇÕES a partir do estado inicial de um episódio, e essa
lista é executada no mesmo ambiente GridWorldCPPEnvV2 que o agente RL usa.
Assim a comparação é maçã-com-maçã: mesma configuração de obstáculos, mesma
posição inicial, mesmas métricas (cobertura, passos).

Convenção de coordenadas (igual à do ambiente):
  ação 0 = (+1, 0) direita   ação 1 = (0, -1) cima
  ação 2 = (-1, 0) esquerda  ação 3 = (0, +1) baixo
"""

from __future__ import annotations

import collections

import numpy as np

DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
ACTION = {(1, 0): 0, (0, -1): 1, (-1, 0): 2, (0, 1): 3}  # (dx,dy) -> ação

ENV_ID = "gymnasium_env/GridWorldCPPv2-v0"


def make_env(size, obstacles, max_steps, local_view=5, render_mode="rgb_array"):
    import gymnasium as gym
    from gymnasium_env.grid_world_cpp_v2 import GridWorldCPPEnvV2
    try:
        gym.register(id=ENV_ID, entry_point=GridWorldCPPEnvV2)
    except Exception:
        pass
    return gym.make(ENV_ID, size=size, obs_quantity=obstacles,
                    max_steps=max_steps, local_view_size=local_view,
                    render_mode=render_mode)


def bfs_path(reachable, src, targets):
    """Menor caminho (lista de passos (dx,dy)) de `src` até a célula mais
    próxima em `targets`, andando só por `reachable`. Retorna [] se src já é
    alvo, ou None se nenhum alvo é alcançável."""
    if src in targets:
        return []
    prev = {src: None}
    q = collections.deque([src])
    found = None
    while q:
        cur = q.popleft()
        if cur in targets:
            found = cur
            break
        for dx, dy in DIRS:
            nb = (cur[0] + dx, cur[1] + dy)
            if nb in reachable and nb not in prev:
                prev[nb] = (cur, (dx, dy))
                q.append(nb)
    if found is None:
        return None
    steps = []
    node = found
    while prev[node] is not None:
        parent, step = prev[node]
        steps.append(step)
        node = parent
    steps.reverse()
    return steps


def path_to_actions(path):
    """Converte lista de passos (dx,dy) em lista de ações inteiras."""
    return [ACTION[step] for step in path]


def execute_actions(env, actions):
    """Executa uma lista de ações num ambiente JÁ RESETADO pelo chamador.
    Retorna (coverage, steps, completed). Para no primeiro done/truncated.
    Se o plano acabar antes de terminar o episódio, conta como não-completo."""
    done = truncated = False
    steps = 0
    info = {"coverage": 0.0}
    for a in actions:
        if done or truncated:
            break
        _, _, done, truncated, info = env.step(int(a))
        steps += 1
    return info["coverage"], steps, (done and not truncated)


def get_state(base):
    """Extrai (reachable, visited, start) do ambiente desempacotado."""
    reachable = set(base._reachable)
    visited = set(base._visited)
    start = (int(base._agent_location[0]), int(base._agent_location[1]))
    return reachable, visited, start


def summarize(name, label, dim, obstacles, max_steps, episodes,
              coverages, steps_list, full):
    """Imprime um resumo padronizado das métricas de um algoritmo."""
    print(f"\n{'='*66}")
    print(f"  {label}")
    print(f"  grid {dim}x{dim} | obstáculos {obstacles} | max_steps {max_steps} "
          f"| {episodes} episódios")
    print(f"{'='*66}")
    print(f"  Taxa de cobertura completa: {full/episodes*100:.1f}% ({full}/{episodes})")
    print(f"  Cobertura média: {np.mean(coverages)*100:.2f}% ± {np.std(coverages)*100:.2f}%")
    print(f"  Passos médios:   {np.mean(steps_list):.1f} ± {np.std(steps_list):.1f}")
    return {
        "algo": name,
        "full_rate": full / episodes,
        "mean_cov": float(np.mean(coverages)),
        "std_cov": float(np.std(coverages)),
        "mean_steps": float(np.mean(steps_list)),
        "std_steps": float(np.std(steps_list)),
    }
