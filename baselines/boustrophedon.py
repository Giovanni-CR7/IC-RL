"""
boustrophedon.py — Cobertura em varredura "vai-e-volta" (zigue-zague).

Aproximação da Boustrophedon Cellular Decomposition (Choset & Pignon, 1998)
citada no referencial. A ideia original decompõe o espaço livre em células sem
obstáculos e faz uma varredura em bumerangue (boustrophedon = "como o boi ara
o campo") dentro de cada uma. Aqui implementamos a varredura zigue-zague
diretamente sobre o grid:

  - percorre as colunas da esquerda para a direita;
  - em cada coluna, sobe ou desce alternadamente (o padrão vai-e-volta);
  - células bloqueadas por obstáculo são puladas;
  - para ir de uma célula-alvo à próxima quando não são adjacentes (por causa
    de um obstáculo ou da troca de coluna), usa BFS para percorrer o caminho
    mais curto pelo espaço livre.

Ao final, faz uma passada de limpeza: se sobrou alguma célula alcançável não
visitada (possível em mapas com obstáculos que quebram a varredura regular),
vai buscá-las via BFS até cobrir tudo. Isso mantém a natureza "varredura
sistemática" mas garante cobertura completa em ambiente conexo.

Expõe:
    plan(base) -> list[int]
"""

from __future__ import annotations

from .common import DIRS, ACTION, bfs_path, get_state


def plan(base):
    reachable, visited, start = get_state(base)
    size = base.size

    # Ordem de varredura boustrophedon: coluna a coluna (x crescente); dentro
    # de cada coluna, y crescente ou decrescente alternando.
    order = []
    for xi, x in enumerate(range(size)):
        ys = range(size) if (xi % 2 == 0) else range(size - 1, -1, -1)
        for y in ys:
            if (x, y) in reachable:
                order.append((x, y))

    pos = start
    actions = []

    def go_to(target):
        nonlocal pos
        path = bfs_path(reachable, pos, {target})
        if path is None:
            return False
        for (dx, dy) in path:
            actions.append(ACTION[(dx, dy)])
            pos = (pos[0] + dx, pos[1] + dy)
            visited.add(pos)
        return True

    # Percorre a ordem de varredura, indo a cada célula ainda não visitada.
    for cell in order:
        if cell in visited:
            continue
        go_to(cell)

    # Limpeza: garante cobertura completa se a varredura deixou lacunas.
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
