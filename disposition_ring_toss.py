"""
Disposition Ring Toss — Tomy-style waterfall ring toss for criminal justice case-flow pressure.

Metaphor:
  Rings = cases
  Water/current = system processing pressure
  Bottom pool = unresolved/open inventory
  Pegs = case dispositions
  Ring colors = modifiers/support conditions

Rate notation (queueing / flow theory):
  λ (lambda) = case generation rate — new cases entering open inventory per second.
  μ (mu)     = successful disposition rate — cases reaching a final peg disposition per second.
  ρ (rho)    = recirculation rate — fraction of case events that return to open inventory
               instead of resolving (missed pegs + peg-level recirculation risk).
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import pygame

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCREEN_W, SCREEN_H = 1280, 720
FPS = 60

POOL_TOP = 580
POOL_MARGIN = 40

PANEL_W = 300
PLAY_W = SCREEN_W - PANEL_W

RING_RADIUS = 14
RING_THICKNESS = 4
PEG_RADIUS = 10
CATCH_RADIUS = 36

# Modifier colors
MODIFIER_COLORS: Dict[str, Tuple[int, int, int]] = {
    "housing": (60, 120, 220),       # Blue
    "treatment": (50, 180, 80),      # Green
    "counsel": (220, 200, 40),       # Yellow
    "acuity": (210, 60, 60),         # Red
    "multi_jurisdiction": (150, 70, 200),  # Purple
    "none": (140, 140, 140),         # Gray
}

MODIFIER_WEIGHTS = {
    "housing": 0.15,
    "treatment": 0.15,
    "counsel": 0.20,
    "acuity": 0.15,
    "multi_jurisdiction": 0.10,
    "none": 0.25,
}

DISPOSITION_NAMES = [
    "Closed: Plea",
    "Closed: Trial",
    "Closed: Dismissal",
    "Closed: Deferred",
    "Treatment Court",
    "Competency Track",
    "Transfer / Other Jurisdiction",
    "Warrant / FTA Loop",
    "Reopened / Recirculated",
]

# Peg index constants for modifier logic
IDX_PLEA, IDX_TRIAL, IDX_DISMISSAL, IDX_DEFERRED = 0, 1, 2, 3
IDX_TREATMENT, IDX_COMPETENCY = 4, 5
IDX_TRANSFER = 6
IDX_WARRANT, IDX_REOPENED = 7, 8

# Probability boosts applied on top of base peg hit probability (clamped 0–1)
MODIFIER_BOOSTS: Dict[str, Dict[int, float]] = {
    "housing": {IDX_DEFERRED: 0.18, IDX_WARRANT: -0.12},
    "treatment": {IDX_TREATMENT: 0.20, IDX_COMPETENCY: 0.18},
    "counsel": {IDX_PLEA: 0.15, IDX_TRIAL: 0.12, IDX_DISMISSAL: 0.14},
    "acuity": {IDX_REOPENED: 0.20, IDX_WARRANT: 0.15},
    "multi_jurisdiction": {IDX_TRANSFER: 0.25},
    "none": {},
}

# Base peg configuration: (base_hit_prob, capacity, processing_delay_sec, recirculation_risk)
PEG_DEFAULTS = [
    (0.22, 8, 2.5, 0.05),   # Plea
    (0.14, 6, 4.0, 0.08),   # Trial
    (0.18, 7, 2.0, 0.04),   # Dismissal
    (0.16, 6, 3.0, 0.10),   # Deferred
    (0.12, 5, 5.0, 0.12),   # Treatment Court
    (0.10, 4, 6.0, 0.15),   # Competency Track
    (0.08, 4, 3.5, 0.10),   # Transfer
    (0.20, 10, 1.5, 0.35),  # Warrant / FTA — high recirculation by design
    (0.15, 8, 2.0, 0.30),   # Reopened / Recirculated
]


class RingState(Enum):
    POOL = auto()
    AIRBORNE = auto()
    PROCESSING = auto()


@dataclass
class Ring:
    """A case moving through the system."""

    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    modifier: str = "none"
    state: RingState = RingState.POOL
    spawn_time: float = 0.0
    ring_id: int = 0
    target_peg: Optional[int] = None
    process_until: float = 0.0
    peg_attempt_cooldown: float = 0.0

    @property
    def color(self) -> Tuple[int, int, int]:
        return MODIFIER_COLORS[self.modifier]

    def launch(self, pressure: float) -> None:
        self.state = RingState.AIRBORNE
        self.vy = -(2.8 + pressure * 2.2)
        self.vx = random.uniform(-1.2, 1.2)

    def reset_to_pool(self, pool_y: float) -> None:
        self.state = RingState.POOL
        self.x = random.uniform(POOL_MARGIN + RING_RADIUS, PLAY_W - POOL_MARGIN - RING_RADIUS)
        self.y = pool_y
        self.vx = 0.0
        self.vy = 0.0
        self.target_peg = None
        self.process_until = 0.0
        self.peg_attempt_cooldown = 0.0


@dataclass
class Peg:
    """A disposition target peg."""

    index: int
    name: str
    x: float
    y: float
    base_hit_prob: float
    capacity: int
    processing_delay: float
    recirculation_risk: float
    occupied: int = 0
    stressed: bool = False
    stress_until: float = 0.0
    disposition_count: int = 0

    @property
    def effective_capacity(self) -> int:
        if self.stressed:
            return max(1, self.capacity // 3)
        return self.capacity

    @property
    def has_room(self) -> bool:
        return self.occupied < self.effective_capacity

    def hit_probability(self, modifier: str) -> float:
        boost = MODIFIER_BOOSTS.get(modifier, {}).get(self.index, 0.0)
        prob = self.base_hit_prob + boost
        if self.stressed:
            prob *= 0.55
        return max(0.02, min(0.95, prob))

    def recirculation_risk_for(self, modifier: str) -> float:
        risk = self.recirculation_risk
        if modifier == "housing" and self.index == IDX_WARRANT:
            risk = max(0.0, risk - 0.15)
        return risk


class MetricsPanel:
    """On-screen metrics display."""

    def __init__(self) -> None:
        self.font = pygame.font.SysFont("consolas", 15)
        self.font_sm = pygame.font.SysFont("consolas", 13)
        self.title_font = pygame.font.SysFont("consolas", 17, bold=True)

    def draw(
        self,
        surface: pygame.Surface,
        sim: "Simulation",
        paused: bool,
        show_help: bool,
    ) -> None:
        panel = pygame.Rect(PLAY_W, 0, PANEL_W, SCREEN_H)
        pygame.draw.rect(surface, (28, 32, 40), panel)
        pygame.draw.line(surface, (70, 80, 100), (PLAY_W, 0), (PLAY_W, SCREEN_H), 2)

        x = PLAY_W + 12
        y = 12
        lines: List[Tuple[str, Tuple[int, int, int]]] = [
            ("CASE-FLOW METRICS", (200, 220, 255)),
            ("", (255, 255, 255)),
            (f"Open inventory: {sim.open_inventory}", (180, 210, 255)),
            (f"Total spawned:  {sim.total_spawned}", (200, 200, 200)),
            (f"Total disposed: {sim.total_disposed}", (140, 220, 140)),
            (f"Recirculated:   {sim.total_recirculated}", (240, 180, 120)),
            ("", (255, 255, 255)),
            (f"λ gen rate:     {sim.lambda_rate:.2f}/s", (255, 220, 120)),
            (f"μ disp rate:    {sim.mu_rate:.2f}/s", (140, 255, 180)),
            (f"ρ recirc rate:  {sim.rho_rate:.1%}", (255, 160, 120)),
            (f"Avg time open:  {sim.avg_time_open:.1f}s", (200, 200, 255)),
            (f"Water pressure: {sim.water_pressure:.2f}", (120, 180, 255)),
            ("", (255, 255, 255)),
            ("DISPOSITION COUNTS", (200, 220, 255)),
        ]

        for text, color in lines:
            if text:
                surf = self.title_font if text.isupper() and ":" not in text else self.font
                surface.blit(surf.render(text, True, color), (x, y))
            y += 22 if text else 8

        for peg in sim.pegs:
            label = f"  {peg.index + 1}. {peg.disposition_count:4d}  {peg.name[:22]}"
            color = (255, 120, 120) if peg.stressed else (170, 170, 170)
            surface.blit(self.font_sm.render(label, True, color), (x, y))
            y += 18

        status = "PAUSED" if paused else "RUNNING"
        surface.blit(self.font.render(status, True, (255, 255, 100)), (x, SCREEN_H - 28))

        if show_help:
            self._draw_help(surface)

    def _draw_help(self, surface: pygame.Surface) -> None:
        overlay = pygame.Surface((PLAY_W, SCREEN_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        surface.blit(overlay, (0, 0))

        font = pygame.font.SysFont("consolas", 16)
        help_lines = [
            "CONTROLS",
            "  UP/DOWN     Water pressure (processing force)",
            "  LEFT/RIGHT  λ spawn rate (case generation)",
            "  SPACE       Pause / resume",
            "  R           Reset simulation",
            "  1-9         Stress-test peg (reduce capacity)",
            "  H           Toggle this help overlay",
            "",
            "METAPHOR",
            "  Rings = cases | Pool = open inventory",
            "  Pegs = dispositions | Colors = modifiers",
            "",
            "RATES",
            "  λ = cases entering open inventory per second",
            "  μ = cases successfully disposed per second",
            "  ρ = share of events returning to open inventory",
            "",
            "RING COLORS",
            "  Blue   Housing support",
            "  Green  Treatment access",
            "  Yellow Strong counsel / readiness",
            "  Red    High acuity / repeat contact",
            "  Purple Multi-jurisdiction",
            "  Gray   No strong modifier",
        ]
        y = 40
        for i, line in enumerate(help_lines):
            color = (255, 220, 100) if i == 0 or line.startswith("  λ") else (230, 230, 230)
            if line in ("METAPHOR", "RATES", "RING COLORS", "CONTROLS"):
                color = (255, 220, 100)
            surface.blit(font.render(line, True, color), (40, y))
            y += 24


class Simulation:
    """Core case-flow simulation."""

    def __init__(self) -> None:
        self.pegs = self._build_pegs()
        self.rings: List[Ring] = []
        self.next_ring_id = 1

        self.lambda_rate = 1.2
        self.water_pressure = 0.5
        self.spawn_accum = 0.0

        self.total_spawned = 0
        self.total_disposed = 0
        self.total_recirculated = 0
        self.total_flow_events = 0  # disposed + recirculated (for ρ)

        self.disposition_times: List[float] = []
        self.mu_window: List[Tuple[float, int]] = []  # (time, disposed delta)
        self._last_mu_count = 0
        self._sim_time = 0.0

        self.pool_center_y = POOL_TOP + (SCREEN_H - POOL_TOP) // 2

    def _build_pegs(self) -> List[Peg]:
        pegs: List[Peg] = []
        count = len(DISPOSITION_NAMES)
        margin = 70
        usable = PLAY_W - 2 * margin
        row_y_top = 140
        row_y_bot = 260
        per_row = (count + 1) // 2
        for i, name in enumerate(DISPOSITION_NAMES):
            row = 0 if i < per_row else 1
            col = i if row == 0 else i - per_row
            cols_in_row = per_row if row == 0 else count - per_row
            x = margin + (col + 0.5) * (usable / cols_in_row)
            y = row_y_top if row == 0 else row_y_bot
            base = PEG_DEFAULTS[i]
            pegs.append(
                Peg(
                    index=i,
                    name=name,
                    x=x,
                    y=y,
                    base_hit_prob=base[0],
                    capacity=base[1],
                    processing_delay=base[2],
                    recirculation_risk=base[3],
                )
            )
        return pegs

    @property
    def open_inventory(self) -> int:
        return sum(1 for r in self.rings if r.state in (RingState.POOL, RingState.AIRBORNE, RingState.PROCESSING))

    @property
    def avg_time_open(self) -> float:
        if not self.disposition_times:
            return 0.0
        return sum(self.disposition_times[-200:]) / len(self.disposition_times[-200:])

    @property
    def mu_rate(self) -> float:
        """μ — successful disposition rate (cases/sec over recent window)."""
        window = 10.0
        cutoff = self._sim_time - window
        self.mu_window = [(t, d) for t, d in self.mu_window if t >= cutoff]
        if len(self.mu_window) < 2:
            return self.total_disposed / max(self._sim_time, 0.1)
        dt = self.mu_window[-1][0] - self.mu_window[0][0]
        dd = self.mu_window[-1][1] - self.mu_window[0][1]
        return dd / max(dt, 0.01)

    @property
    def rho_rate(self) -> float:
        """ρ — recirculation rate: fraction of flow events that return to inventory."""
        if self.total_flow_events == 0:
            return 0.0
        return self.total_recirculated / self.total_flow_events

    def reset(self) -> None:
        self.__init__()

    def _pick_modifier(self) -> str:
        keys = list(MODIFIER_WEIGHTS.keys())
        weights = [MODIFIER_WEIGHTS[k] for k in keys]
        return random.choices(keys, weights=weights, k=1)[0]

    def _spawn_ring(self) -> None:
        ring = Ring(
            x=random.uniform(POOL_MARGIN + RING_RADIUS, PLAY_W - POOL_MARGIN - RING_RADIUS),
            y=self.pool_center_y + random.uniform(-20, 20),
            modifier=self._pick_modifier(),
            spawn_time=self._sim_time,
            ring_id=self.next_ring_id,
        )
        self.next_ring_id += 1
        self.rings.append(ring)
        self.total_spawned += 1

    def stress_peg(self, index: int) -> None:
        if 0 <= index < len(self.pegs):
            peg = self.pegs[index]
            peg.stressed = True
            peg.stress_until = self._sim_time + 8.0

    def update(self, dt: float) -> None:
        self._sim_time += dt

        # λ — spawn new cases into open inventory at configurable rate
        self.spawn_accum += self.lambda_rate * dt
        while self.spawn_accum >= 1.0:
            self.spawn_accum -= 1.0
            self._spawn_ring()

        # Launch pooled rings when water pressure pushes them upward
        pool_ready = [r for r in self.rings if r.state == RingState.POOL]
        launch_chance = min(0.35, 0.08 + self.water_pressure * 0.12) * dt * max(1, len(pool_ready) * 0.3)
        for ring in pool_ready:
            if random.random() < launch_chance:
                ring.launch(self.water_pressure)

        for peg in self.pegs:
            if peg.stressed and self._sim_time >= peg.stress_until:
                peg.stressed = False

        for ring in list(self.rings):
            self._update_ring(ring, dt)

        # Track μ over sliding window
        if self.total_disposed != self._last_mu_count:
            self.mu_window.append((self._sim_time, self.total_disposed))
            self._last_mu_count = self.total_disposed

    def _update_ring(self, ring: Ring, dt: float) -> None:
        if ring.state == RingState.PROCESSING:
            if self._sim_time >= ring.process_until:
                self._finish_processing(ring)
            return

        if ring.state == RingState.POOL:
            ring.y += math.sin(self._sim_time * 2 + ring.ring_id) * 0.15
            return

        # AIRBORNE — water jet + turbulence
        ring.peg_attempt_cooldown = max(0.0, ring.peg_attempt_cooldown - dt)

        accel_up = -(0.4 + self.water_pressure * 1.6)
        ring.vy += accel_up * dt
        ring.vy += random.uniform(-0.6, 0.6) * dt * (0.5 + self.water_pressure)
        ring.vx += random.uniform(-1.8, 1.8) * dt
        ring.vx *= 0.985
        ring.vy *= 0.992

        ring.x += ring.vx * dt * 60
        ring.y += ring.vy * dt * 60

        ring.x = max(RING_RADIUS, min(PLAY_W - RING_RADIUS, ring.x))

        # Peg catch attempts while rising through peg zone
        if ring.vy < 0 and ring.peg_attempt_cooldown <= 0:
            for peg in self.pegs:
                dist = math.hypot(ring.x - peg.x, ring.y - peg.y)
                if dist < CATCH_RADIUS and peg.has_room:
                    prob = peg.hit_probability(ring.modifier)
                    if random.random() < prob * dt * 8:
                        self._start_processing(ring, peg)
                        return

        # Missed the peg field — recirculate back to open inventory
        if ring.y < 80:
            self._recirculate(ring, reason="missed_peg_field")
        elif ring.y > SCREEN_H - 30:
            ring.reset_to_pool(self.pool_center_y)

    def _start_processing(self, ring: Ring, peg: Peg) -> None:
        ring.state = RingState.PROCESSING
        ring.target_peg = peg.index
        ring.process_until = self._sim_time + peg.processing_delay
        ring.vx = 0.0
        ring.vy = 0.0
        ring.x = peg.x
        ring.y = peg.y
        peg.occupied += 1

    def _finish_processing(self, ring: Ring) -> None:
        peg = self.pegs[ring.target_peg]  # type: ignore[index]
        peg.occupied = max(0, peg.occupied - 1)

        risk = peg.recirculation_risk_for(ring.modifier)
        if random.random() < risk:
            self._recirculate(ring, reason="peg_recirculation")
        else:
            self._dispose(ring, peg)

    def _dispose(self, ring: Ring, peg: Peg) -> None:
        peg.disposition_count += 1
        self.total_disposed += 1
        self.total_flow_events += 1
        self.disposition_times.append(self._sim_time - ring.spawn_time)
        self.rings.remove(ring)

    def _recirculate(self, ring: Ring, reason: str) -> None:
        self.total_recirculated += 1
        self.total_flow_events += 1
        ring.reset_to_pool(self.pool_center_y)
        # Slight modifier drift on recirculation (case complexity evolves)
        if random.random() < 0.25:
            ring.modifier = self._pick_modifier()

    def draw(self, surface: pygame.Surface) -> None:
        # Background gradient feel
        surface.fill((18, 28, 48))

        # Waterfall / current column
        current_rect = pygame.Rect(PLAY_W // 2 - 60, 0, 120, SCREEN_H)
        s = pygame.Surface((120, SCREEN_H), pygame.SRCALPHA)
        for row in range(0, SCREEN_H, 4):
            alpha = 20 + int(15 * math.sin(row * 0.05 + self._sim_time * 3))
            pygame.draw.rect(s, (80, 160, 220, alpha), (0, row, 120, 4))
        surface.blit(s, current_rect.topleft)

        # Upward flow lines
        for i in range(12):
            phase = (self._sim_time * 80 + i * 55) % SCREEN_H
            x = PLAY_W // 2 + math.sin(self._sim_time * 2 + i) * 30
            drift = math.sin(self._sim_time * 3 + i * 1.7) * 8
            pygame.draw.line(
                surface,
                (100, 180, 230, 80),
                (x, SCREEN_H - phase),
                (x + drift, SCREEN_H - phase - 40),
                1,
            )

        # Bottom pool = open inventory
        pool_rect = pygame.Rect(POOL_MARGIN, POOL_TOP, PLAY_W - 2 * POOL_MARGIN, SCREEN_H - POOL_TOP - 10)
        pygame.draw.rect(surface, (25, 55, 90), pool_rect, border_radius=8)
        pygame.draw.rect(surface, (70, 130, 190), pool_rect, 2, border_radius=8)

        font = pygame.font.SysFont("consolas", 14)
        surface.blit(font.render("OPEN INVENTORY (bottom pool)", True, (160, 200, 255)), (POOL_MARGIN + 8, POOL_TOP + 8))

        # Pegs
        label_font = pygame.font.SysFont("consolas", 11)
        for peg in self.pegs:
            color = (255, 200, 80) if peg.stressed else (220, 180, 60)
            pygame.draw.circle(surface, (60, 50, 30), (int(peg.x), int(peg.y)), PEG_RADIUS + 2)
            pygame.draw.circle(surface, color, (int(peg.x), int(peg.y)), PEG_RADIUS)
            cap = f"{peg.occupied}/{peg.effective_capacity}"
            label = peg.name if len(peg.name) <= 24 else peg.name[:22] + ".."
            surface.blit(label_font.render(label, True, (220, 220, 220)), (int(peg.x) - 55, int(peg.y) - 32))
            surface.blit(label_font.render(cap, True, (180, 180, 180)), (int(peg.x) - 12, int(peg.y) + 14))

        # Rings
        for ring in self.rings:
            cx, cy = int(ring.x), int(ring.y)
            pygame.draw.circle(surface, ring.color, (cx, cy), RING_RADIUS, RING_THICKNESS)
            if ring.state == RingState.PROCESSING:
                pygame.draw.circle(surface, (255, 255, 255), (cx, cy), 3)


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Disposition Ring Toss — Case-Flow Pressure Prototype")
    clock = pygame.time.Clock()

    sim = Simulation()
    panel = MetricsPanel()
    paused = False
    show_help = False

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    sim = Simulation()
                elif event.key == pygame.K_h:
                    show_help = not show_help
                elif event.key == pygame.K_UP:
                    sim.water_pressure = min(1.5, sim.water_pressure + 0.05)
                elif event.key == pygame.K_DOWN:
                    sim.water_pressure = max(0.05, sim.water_pressure - 0.05)
                elif event.key == pygame.K_RIGHT:
                    sim.lambda_rate = min(8.0, sim.lambda_rate + 0.1)
                elif event.key == pygame.K_LEFT:
                    sim.lambda_rate = max(0.1, sim.lambda_rate - 0.1)
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    sim.stress_peg(event.key - pygame.K_1)

        if not paused:
            sim.update(dt)

        sim.draw(screen)
        panel.draw(screen, sim, paused, show_help)
        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
