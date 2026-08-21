"""
frontier.py — Cobertura por exploração de fronteira (frontier-based).

Este é o método de exploração usado na prática em robótica real (veículos
autônomos de superfície, drones de busca e resgate, exploração com SLAM): o
agente "expande" a cobertura a partir da sua posição, indo sempre à FRONTEIRA
mais próxima — a borda entre a região já coberta e a ainda não visitada —,
contornando obstáculos. Visualmente parece uma mancha que cresce a partir do
agente, como uma minhoca que se estende preenchendo o espaço livre.

Diferença conceitual em relação aos outros baselines: enquanto BFS/STC/
Boustrophedon planejam a rota inteira de antemão com o mapa completo, o
frontier-based decide passo a passo com base na fronteira atual. Isso o torna
o análogo mais direto do agente RL de observação parcial — a diferença é que
aqui a regra de decisão é fixa (guloso rumo à fronteira mais próxima), não
aprendida. É também exatamente o sinal de direção-objetivo que embutimos na
observação do ambiente, aqui usado como POLÍTICA completa.

Implementação: a cada passo, BFS multi-fonte (wavefront) a partir de todas as
células de fronteira gera um campo de distância; o agente move-se para o
vizinho com menor distância (primeiro passo do caminho mais curto até a
fronteira mais próxima). Cobre 100% em ambiente conexo.

Expõe:
    plan(base) -> list[int]
"""

from __future__ import annotations

import collections

from .common import DIRS, ACTION, get_state


def _frontier_step(reachable, visited, pos):
    """Retorna a ação rumo à fronteira não visitada mais próxima (wavefront),
    ou None se não há fronteira."""
    frontier = reachable - visited
    if not frontier:
        return None

    dist = {c: 0 for c in frontier}
    q = collections.deque(frontier)
    while q:
        cx, cy = q.popleft()
        d = dist[(cx, cy)]
        for dx, dy in DIRS:
            nb = (cx + dx, cy + dy)
            if nb in reachable and nb not in dist:
                dist[nb] = d + 1
                q.append(nb)

    best_dir, best_d = None, dist.get(pos, 0)
    for dx, dy in DIRS:
        nd = dist.get((pos[0] + dx, pos[1] + dy))
        if nd is not None and nd < best_d:
            best_d, best_dir = nd, (dx, dy)

    if best_dir is None:
        return None
    return ACTION[best_dir]


def plan(base):
    reachable, visited, start = get_state(base)
    pos = start
    actions = []

    guard = 0
    max_guard = len(reachable) * 6 + 10
    while visited != reachable and guard < max_guard:
        guard += 1
        frontier = reachable - visited
        if not frontier:
            break

        # campo de distância wavefront a partir da fronteira
        dist = {c: 0 for c in frontier}
        q = collections.deque(frontier)
        while q:
            cx, cy = q.popleft()
            d = dist[(cx, cy)]
            for dx, dy in DIRS:
                nb = (cx + dx, cy + dy)
                if nb in reachable and nb not in dist:
                    dist[nb] = d + 1
                    q.append(nb)

        best_dir, best_d = None, dist.get(pos, 0)
        for dx, dy in DIRS:
            nd = dist.get((pos[0] + dx, pos[1] + dy))
            if nd is not None and nd < best_d:
                best_d, best_dir = nd, (dx, dy)

        if best_dir is None:
            break
        actions.append(ACTION[best_dir])
        pos = (pos[0] + best_dir[0], pos[1] + best_dir[1])
        visited.add(pos)

    return actions
