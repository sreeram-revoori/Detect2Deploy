"""
scenario_generator.py
Procedurally generates synthetic AV driving scenarios as bird's-eye-view frames
with ground-truth bounding boxes. Each scenario represents a single timestep
snapshot of the environment around the ego vehicle.
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Tuple
import random

# ── Canvas / world config ─────────────────────────────────────────────────────
FRAME_W, FRAME_H = 640, 640          # pixels
WORLD_RANGE       = 50.0             # metres represented by half the canvas
SCALE             = FRAME_W / (2 * WORLD_RANGE)   # px per metre

# ── Object class palette (BGR) ────────────────────────────────────────────────
CLASS_CONFIG = {
    "vehicle":     {"color": (0,   200, 255), "id": 0, "w_range": (3.5, 5.5), "h_range": (1.8, 2.5)},
    "pedestrian":  {"color": (0,   255,  80), "id": 1, "w_range": (0.5, 0.9), "h_range": (0.5, 0.9)},
    "cyclist":     {"color": (255, 140,   0), "id": 2, "w_range": (0.8, 1.2), "h_range": (1.5, 2.0)},
    "cone":        {"color": (0,    80, 255), "id": 3, "w_range": (0.3, 0.5), "h_range": (0.3, 0.5)},
}

WEATHER_CONDITIONS = ["clear", "rain", "fog", "night"]


@dataclass
class BBox:
    """Axis-aligned bounding box in pixel space."""
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int
    class_name: str
    confidence: float = 1.0          # ground-truth = 1.0

    @property
    def xywh(self) -> Tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1

    @property
    def xyxy(self) -> Tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


@dataclass
class Scenario:
    scenario_id: int
    weather: str
    num_objects: int
    frame: np.ndarray                # HxWx3 uint8 BGR
    ground_truth: List[BBox] = field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _world_to_pixel(x_m: float, y_m: float) -> Tuple[int, int]:
    """Convert world coordinates (metres, ego-centred) to pixel coordinates."""
    px = int(FRAME_W / 2 + x_m * SCALE)
    py = int(FRAME_H / 2 - y_m * SCALE)   # y-axis flipped for image coords
    return px, py


def _draw_road(canvas: np.ndarray, weather: str) -> None:
    """Draw a simple T-intersection road layout."""
    road_color = {
        "clear": (60, 60, 60),
        "rain":  (50, 50, 55),
        "fog":   (80, 80, 85),
        "night": (30, 30, 30),
    }[weather]

    # Main straight road (vertical)
    cv2.rectangle(canvas,
                  (FRAME_W//2 - 60, 0),
                  (FRAME_W//2 + 60, FRAME_H),
                  road_color, -1)
    # Cross road (horizontal, upper half)
    cv2.rectangle(canvas,
                  (0, FRAME_H//4 - 60),
                  (FRAME_W, FRAME_H//4 + 60),
                  road_color, -1)

    # Lane markings
    line_color = (200, 200, 200)
    for y in range(0, FRAME_H, 40):
        cv2.line(canvas, (FRAME_W//2, y), (FRAME_W//2, y + 20), line_color, 1)
    for x in range(0, FRAME_W, 40):
        cv2.line(canvas, (x, FRAME_H//4), (x + 20, FRAME_H//4), line_color, 1)


def _apply_weather_fx(canvas: np.ndarray, weather: str) -> np.ndarray:
    if weather == "rain":
        noise = np.random.randint(0, 30, canvas.shape, dtype=np.uint8)
        canvas = cv2.add(canvas, noise)
        # Rain streaks
        for _ in range(120):
            x = random.randint(0, FRAME_W)
            y = random.randint(0, FRAME_H)
            cv2.line(canvas, (x, y), (x - 2, y + 12), (180, 180, 200), 1)
    elif weather == "fog":
        fog_layer = np.full_like(canvas, 160)
        canvas = cv2.addWeighted(canvas, 0.55, fog_layer, 0.45, 0)
    elif weather == "night":
        canvas = (canvas * 0.35).astype(np.uint8)
        # Headlight cone from ego
        cx, cy = FRAME_W // 2, FRAME_H // 2
        overlay = canvas.copy()
        pts = np.array([[cx, cy], [cx - 80, cy - 180], [cx + 80, cy - 180]])
        cv2.fillPoly(overlay, [pts], (60, 60, 40))
        canvas = cv2.addWeighted(canvas, 0.5, overlay, 0.5, 0)
    return canvas


def _place_object(canvas: np.ndarray,
                  class_name: str,
                  rng: np.random.Generator) -> BBox:
    """Randomly place one object on the canvas; return its pixel BBox."""
    cfg = CLASS_CONFIG[class_name]

    # World position (metres, avoiding ego centre ±4 m)
    while True:
        x_m = rng.uniform(-WORLD_RANGE + 5, WORLD_RANGE - 5)
        y_m = rng.uniform(-WORLD_RANGE + 5, WORLD_RANGE - 5)
        if abs(x_m) > 4 or abs(y_m) > 4:
            break

    w_m = rng.uniform(*cfg["w_range"])
    h_m = rng.uniform(*cfg["h_range"])

    cx, cy = _world_to_pixel(x_m, y_m)
    half_w = int(w_m * SCALE / 2)
    half_h = int(h_m * SCALE / 2)

    x1 = max(0, cx - half_w)
    y1 = max(0, cy - half_h)
    x2 = min(FRAME_W - 1, cx + half_w)
    y2 = min(FRAME_H - 1, cy + half_h)

    # Draw filled rect + border
    cv2.rectangle(canvas, (x1, y1), (x2, y2), cfg["color"], -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 1)

    # Class label
    cv2.putText(canvas, class_name[0].upper(),
                (x1 + 2, y2 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

    return BBox(x1=x1, y1=y1, x2=x2, y2=y2,
                class_id=cfg["id"], class_name=class_name)


def _draw_ego(canvas: np.ndarray) -> None:
    cx, cy = FRAME_W // 2, FRAME_H // 2
    cv2.rectangle(canvas, (cx - 15, cy - 25), (cx + 15, cy + 25), (0, 0, 255), -1)
    cv2.rectangle(canvas, (cx - 15, cy - 25), (cx + 15, cy + 25), (255, 255, 255), 1)
    cv2.putText(canvas, "EGO", (cx - 14, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)


# ── Public API ────────────────────────────────────────────────────────────────

def generate_scenario(scenario_id: int,
                      seed: int | None = None,
                      weather: str | None = None,
                      num_objects: int | None = None) -> Scenario:
    """
    Generate one synthetic BEV scenario frame with ground-truth bounding boxes.

    Args:
        scenario_id : unique integer identifier
        seed        : RNG seed for reproducibility
        weather     : one of WEATHER_CONDITIONS; random if None
        num_objects : total objects to place; random 3–12 if None
    Returns:
        Scenario dataclass
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    weather     = weather     or rng.choice(WEATHER_CONDITIONS)
    num_objects = num_objects or int(rng.integers(3, 13))

    # Background (grass/dirt)
    bg_color = {"clear": (34,139,34), "rain": (28,115,28),
                "fog": (100,120,100), "night": (15,60,15)}[weather]
    canvas = np.full((FRAME_H, FRAME_W, 3), bg_color, dtype=np.uint8)

    _draw_road(canvas, weather)

    # Place objects with weighted class distribution
    class_weights = [0.50, 0.25, 0.15, 0.10]   # vehicle, ped, cyclist, cone
    class_names   = list(CLASS_CONFIG.keys())
    chosen_classes = rng.choice(class_names, size=num_objects, p=class_weights)

    gt_boxes: List[BBox] = []
    for cls in chosen_classes:
        bbox = _place_object(canvas, cls, rng)
        gt_boxes.append(bbox)

    canvas = _apply_weather_fx(canvas, weather)
    _draw_ego(canvas)

    # HUD overlay
    cv2.putText(canvas, f"ID:{scenario_id}  WX:{weather}  OBJ:{num_objects}",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)

    return Scenario(
        scenario_id=scenario_id,
        weather=weather,
        num_objects=num_objects,
        frame=canvas,
        ground_truth=gt_boxes,
    )


def generate_batch(n: int,
                   base_seed: int = 42) -> List[Scenario]:
    """Generate n scenarios deterministically."""
    return [generate_scenario(i, seed=base_seed + i) for i in range(n)]
