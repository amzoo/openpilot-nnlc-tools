from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Frame:
    timestamp: float
    steering_torque: float
    torque_output: float
    actual_lateral_accel: float
    desired_lateral_accel: float
    steering_angle_deg: float
    steering_rate_deg: float
    v_ego: float
    a_ego: float
    steering_pressed: bool
    lane_change_state: int = 0


@dataclass
class Event:
    """A contiguous window of frames where steering_pressed is True."""
    frames: list  # list[Frame]
    start_idx: int
    end_idx: int

    @property
    def duration_s(self) -> float:
        if len(self.frames) < 2:
            return len(self.frames) * 0.01
        return self.frames[-1].timestamp - self.frames[0].timestamp

    @property
    def n_frames(self) -> int:
        return len(self.frames)


@dataclass
class ClassificationResult:
    label: str          # "driver" or "mechanical"
    confidence: float   # 0.0–1.0
    mechanical_score: float
    driver_score: float
    stage: int          # 1, 2, or 3 — which cascade stage made the decision
    features: dict = field(default_factory=dict)
