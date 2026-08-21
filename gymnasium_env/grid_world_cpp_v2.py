from __future__ import annotations

import collections
from typing import Optional

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces


class GridWorldCPPEnvV2(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    def __init__(
        self,
        size: int = 10,
        obs_quantity: int = 12,
        max_steps: int = 500,
        local_view_size: int = 5,
        include_goal_direction: bool = True,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        assert local_view_size % 2 == 1, "local_view_size deve ser ímpar (ex: 3, 5, 7)"

        self.size = int(size)
        self.obs_quantity = int(obs_quantity)
        self.max_steps = int(max_steps)
        self.local_view_size = int(local_view_size)
        self.include_goal_direction = bool(include_goal_direction)
        self.render_mode = render_mode
        self.window_size = 512

        # Estado interno
        self._agent_location: np.ndarray = np.array([0, 0], dtype=int)
        self._obstacles: set[tuple[int, int]] = set()
        self._visited: set[tuple[int, int]] = set()
        self._reachable: set[tuple[int, int]] = set()
        self.count_steps: int = 0

        # Espaço de ação: direita, cima, esquerda, baixo
        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1,  0]),   # direita
            1: np.array([0, -1]),   # cima
            2: np.array([-1, 0]),   # esquerda
            3: np.array([0,  1]),   # baixo
        }
        self._bfs_dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        lv = self.local_view_size
        if self.include_goal_direction:
            global_low = np.array([0.0, 0.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32)
            global_high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        else:
            global_low = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            global_high = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        self.observation_space = spaces.Dict({
            "local": spaces.Box(
                low=0.0, high=1.0,
                shape=(lv * lv,),
                dtype=np.float32,
            ),
            "global": spaces.Box(low=global_low, high=global_high, dtype=np.float32),
        })

        self.window = None
        self.clock = None

    # ── propriedades ──────────────────────────────────────────────────────────

    @property
    def coverage_ratio(self) -> float:
        total = len(self._reachable)
        return len(self._visited) / total if total > 0 else 1.0

    # ── direção-objetivo (BFS multi-fonte / wavefront) ─────────────────────────

    def _compute_goal_direction(self) -> tuple[float, float]:
        """
        Calcula a direção (dx, dy) até a célula não visitada alcançável mais
        próxima, considerando obstáculos.
        """
        frontier = self._reachable - self._visited
        if not frontier:
            return 0.0, 0.0

        # BFS multi-fonte: todas as células de fronteira começam com dist=0
        dist: dict[tuple[int, int], int] = {c: 0 for c in frontier}
        queue = collections.deque(frontier)
        while queue:
            cx, cy = queue.popleft()
            d = dist[(cx, cy)]
            for dx, dy in self._bfs_dirs:
                nb = (cx + dx, cy + dy)
                if nb in self._reachable and nb not in dist:
                    dist[nb] = d + 1
                    queue.append(nb)

        ax, ay = int(self._agent_location[0]), int(self._agent_location[1])
        agent_pos = (ax, ay)
        agent_dist = dist.get(agent_pos)
        if agent_dist is None:
            # não deveria ocorrer (agent_pos é sempre alcançável), mas evita
            # crash em caso de estado inesperado
            return 0.0, 0.0

        best_dir = (0, 0)
        best_d = agent_dist
        for dx, dy in self._bfs_dirs:
            nb = (ax + dx, ay + dy)
            nd = dist.get(nb)
            if nd is not None and nd < best_d:
                best_d = nd
                best_dir = (dx, dy)

        return float(best_dir[0]), float(best_dir[1])

    # ── observação ────────────────────────────────────────────────────────────

    def _build_local_view(self) -> np.ndarray:
        
        half = self.local_view_size // 2
        ax, ay = self._agent_location
        view = np.zeros((self.local_view_size, self.local_view_size), dtype=np.float32)
        for di in range(-half, half + 1):
            for dj in range(-half, half + 1):
                nx, ny = ax + dj, ay + di
                vi, vj = di + half, dj + half
                if not (0 <= nx < self.size and 0 <= ny < self.size):
                    view[vi, vj] = 1.0
                elif (nx, ny) in self._obstacles:
                    view[vi, vj] = 1.0
                elif (nx, ny) in self._visited:
                    view[vi, vj] = 0.5
                # else: 0.0 (livre, não visitado)
        return view.flatten()

    def _get_obs(self) -> dict:
        ax, ay = self._agent_location
        span = max(1, self.size - 1)
        steps_remaining = max(0, self.max_steps - self.count_steps) / self.max_steps

        global_vals = [
            ax / span,
            ay / span,
            self.coverage_ratio,
            steps_remaining,
        ]
        if self.include_goal_direction:
            dx, dy = self._compute_goal_direction()
            global_vals += [dx, dy]

        return {
            "local": self._build_local_view(),
            "global": np.array(global_vals, dtype=np.float32),
        }

    def _get_info(self) -> dict:
        return {
            "coverage": self.coverage_ratio,
            "visited_cells": len(self._visited),
            "total_free_cells": len(self._reachable),
            "steps": self.count_steps,
        }

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.count_steps = 0
        self._obstacles = set()
        self._visited = set()
        self._reachable = set()

        # Posição inicial aleatória
        ax = int(self.np_random.integers(0, self.size))
        ay = int(self.np_random.integers(0, self.size))
        self._agent_location = np.array([ax, ay], dtype=int)
        start = (ax, ay)

        # Gera obstáculos em posições livres (nunca sobre o agente)
        while len(self._obstacles) < self.obs_quantity:
            ox = int(self.np_random.integers(0, self.size))
            oy = int(self.np_random.integers(0, self.size))
            pos = (ox, oy)
            if pos != start and pos not in self._obstacles:
                self._obstacles.add(pos)

        # BFS: calcula células alcançáveis a partir do agente
        queue = collections.deque([start])
        self._reachable = {start}
        while queue:
            cx, cy = queue.popleft()
            for dx, dy in self._bfs_dirs:
                nb = (cx + dx, cy + dy)
                nx, ny = nb
                if (0 <= nx < self.size and 0 <= ny < self.size
                        and nb not in self._obstacles
                        and nb not in self._reachable):
                    self._reachable.add(nb)
                    queue.append(nb)

        self._visited.add(start)

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return obs, info

    # ── step ──────────────────────────────────────────────────────────────────

    def step(self, action: int):
        action = int(action)
        direction = self._action_to_direction[action]
        new_location = self._agent_location + direction
        nx, ny = new_location

        # Verifica se o movimento é válido
        in_bounds = 0 <= nx < self.size and 0 <= ny < self.size
        not_obstacle = (nx, ny) not in self._obstacles

        stayed = False
        if in_bounds and not_obstacle:
            self._agent_location = new_location
        else:
            stayed = True  # bateu na parede ou obstáculo

        current_pos = tuple(self._agent_location)
        is_new_cell = current_pos not in self._visited

        # --- Reward ---
        reward = -0.1  # passo base

        if stayed:
            reward -= 0.5  # penalidade de colisão
        elif is_new_cell:
            reward += 1.0
            self._visited.add(current_pos)
        else:
            reward -= 0.3  # penalidade de revisita

        self.count_steps += 1

        # Término por cobertura completa
        full_coverage = self._visited >= self._reachable
        terminated = full_coverage
        if full_coverage:
            reward += 10.0

        # Truncamento por limite de passos
        truncated = not terminated and self.count_steps >= self.max_steps
        if truncated:
            reward -= 5.0

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, truncated, info

    # ── render ────────────────────────────────────────────────────────────────

    def render(self):
        if self.render_mode in ("human", "rgb_array"):
            return self._render_frame()

    def _render_frame(self):
        if self.render_mode is None:
            return

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption(f"CPP v3 — {self.size}×{self.size}")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((245, 245, 245))
        pix = self.window_size / self.size

        # Células alcançáveis não visitadas (cinza claro)
        for rx, ry in self._reachable:
            if (rx, ry) not in self._visited:
                pygame.draw.rect(canvas, (210, 210, 210),
                                 pygame.Rect(rx * pix, ry * pix, pix, pix))

        # Células visitadas (verde)
        for vx, vy in self._visited:
            pygame.draw.rect(canvas, (144, 238, 144),
                             pygame.Rect(vx * pix, vy * pix, pix, pix))

        # Obstáculos (preto)
        for ox, oy in self._obstacles:
            pygame.draw.rect(canvas, (30, 30, 30),
                             pygame.Rect(ox * pix, oy * pix, pix, pix))

        # Agente (azul)
        ax, ay = self._agent_location
        pygame.draw.circle(canvas, (0, 80, 220),
                           ((ax + 0.5) * pix, (ay + 0.5) * pix), pix / 3)

        if self.include_goal_direction:
            dx, dy = self._compute_goal_direction()
            if dx != 0 or dy != 0:
                cx, cy = (ax + 0.5) * pix, (ay + 0.5) * pix
                ex, ey = cx + dx * pix * 0.8, cy + dy * pix * 0.8
                pygame.draw.line(canvas, (220, 20, 20), (cx, cy), (ex, ey), 3)

        # Janela local 
        half = self.local_view_size // 2
        lx = (ax - half) * pix
        ly = (ay - half) * pix
        lw = lh = self.local_view_size * pix
        pygame.draw.rect(canvas, (255, 140, 0), pygame.Rect(lx, ly, lw, lh), 2)

        # Grid lines
        for i in range(self.size + 1):
            pygame.draw.line(canvas, (180, 180, 180),
                             (0, i * pix), (self.window_size, i * pix), 1)
            pygame.draw.line(canvas, (180, 180, 180),
                             (i * pix, 0), (i * pix, self.window_size), 1)

        # HUD
        if pygame.get_init():
            font = pygame.font.SysFont(None, 22)
            hud = font.render(
                f"Coverage: {self.coverage_ratio:.1%}  |  Steps: {self.count_steps}/{self.max_steps}",
                True, (20, 20, 20)
            )
            canvas.blit(hud, (5, 5))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
