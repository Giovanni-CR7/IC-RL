"""
spanning_tree.py — Cobertura por árvore geradora (Spanning Tree Coverage).

Aproximação do Spanning Tree Coverage (Gabriely & Rimon, 2001) citado no
referencial. A ideia: construir uma árvore geradora sobre as células livres do
grid e percorrê-la de forma que todas as células sejam visitadas.

Implementação:
  - constrói uma spanning tree das células alcançáveis via DFS a partir da
    posição inicial (cada célula alcançável vira nó; arestas conectam vizinhos
    livres);
  - percorre a árvore em DFS (pré-ordem). Como células consecutivas na
    travessia DFS nem sempre são adjacentes no grid (ao "voltar" de um ramo),
    usa BFS para percorrer o caminho mais curto entre elas.

A travessia de árvore garante que toda célula é visitada ao menos uma vez, ou
seja, cobertura completa em ambiente conexo. O padrão resultante é a varredura
sistemática característica do STC.

Expõe:
    plan(base) -> list[int]
"""

from __future__ import annotations

from .common import DIRS, ACTION, bfs_path, get_state


def _build_spanning_tree(reachable, start):
    """DFS a partir de start; retorna dict filho->pai e a ordem de visita
    (pré-ordem) dos nós."""
    visited = {start}
    order = [start]
    stack = [start]
    parent = {start: None}
    while stack:
        cur = stack[-1]
        advanced = False
        for dx, dy in DIRS:
            nb = (cur[0] + dx, cur[1] + dy)
            if nb in reachable and nb not in visited:
                visited.add(nb)
                parent[nb] = cur
                order.append(nb)
                stack.append(nb)
                advanced = True
                break
        if not advanced:
            stack.pop()
    return parent, order


def plan(base):
    reachable, visited, start = get_state(base)
    _, order = _build_spanning_tree(reachable, start)

    pos = start
    actions = []

    for cell in order:
        if cell == start:
            continue
        # anda até a próxima célula da travessia (adjacente ou via BFS)
        path = bfs_path(reachable, pos, {cell})
        if path is None:
            continue
        for (dx, dy) in path:
            actions.append(ACTION[(dx, dy)])
            pos = (pos[0] + dx, pos[1] + dy)
            visited.add(pos)

    # Limpeza defensiva (a travessia já cobre tudo, mas garante).
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
