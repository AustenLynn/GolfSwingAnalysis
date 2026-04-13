from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class ImuSampleIn(BaseModel):
    t_ms: float
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    yaw: float
    pitch: float
    roll: float


class AnalyzeRequest(BaseModel):
    session_id: str
    captured_at: str
    label: Optional[int] = None
    samples: list[ImuSampleIn]


class LabelRequest(BaseModel):
    label: int  # 0 or 1


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class FeatureImportanceOut(BaseModel):
    feature: str
    importance: float
    value: float
    verdict: str   # "good" | "bad" | "neutral"


class PhaseTimingOut(BaseModel):
    t_start_ms: float
    t_top_ms: float
    t_impact_ms: float
    t_end_ms: float
    T_backswing_s: float
    T_downswing_s: float
    tempo_ratio: float


class SwingResultOut(BaseModel):
    id: Optional[str] = None
    swing_index: int
    verdict: str          # "GOOD" | "BAD"
    label: int            # 1 | 0
    confidence: float
    features: dict        # all 19 extracted features
    feature_importances: list[FeatureImportanceOut]
    phase_timing: PhaseTimingOut


class AnalyzeResponse(BaseModel):
    session_id: str
    swings_detected: int
    swings: list[SwingResultOut]


class SwingHistoryEntry(BaseModel):
    id: str
    session_id: str
    swing_index: int
    captured_at: str
    verdict: str
    label: int
    user_label: Optional[int]
    confidence: float
    acc_mean: Optional[float]
    omega_mean: Optional[float]
    tempo_ratio: Optional[float]


class SwingsListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    swings: list[SwingHistoryEntry]


class LabelResponse(BaseModel):
    id: str
    label: int
    updated_at: str


class RetrainResponse(BaseModel):
    status: str
    loocv_accuracy: float
    n_good: int
    n_bad: int
    model_type: str


class ModelInfoResponse(BaseModel):
    model_type: str
    model_class: str
    features: list[str]
    loocv_accuracy: Optional[float]
    n_training_samples: int
    n_good: int
    n_bad: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    db_connected: bool
