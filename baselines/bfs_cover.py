"""
bfs_cover.py — Cobertura baseada em Breadth-First Search (BFS).

Algoritmo clássico citado no referencial (busca em grafos). A estratégia:
enquanto houver célula não visitada, calcula o caminho mais curto (BFS) da
posição atual até a célula não visitada mais próxima e anda até lá, marcando
como visitadas todas as células do trajeto. Repete até cobrir tudo.
"""

from __future__ import annotations

from .common import DIRS, ACTION, bfs_path, get_state


def plan(base):
    reachable, visited, start = get_state(base)
    pos = start
    actions = []

    guard = 0
    max_guard = len(reachable) * 4 + 10
    while visited != reachable and guard < max_guard:
        guard += 1
        unvisited = reachable - visited
        path = bfs_path(reachable, pos, unvisited)
        if not path:
            break
        for (dx, dy) in path:
            actions.append(ACTION[(dx, dy)])
            pos = (pos[0] + dx, pos[1] + dy)
            visited.add(pos)
    return actions
