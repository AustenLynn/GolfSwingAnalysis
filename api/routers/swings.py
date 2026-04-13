import json
from datetime import datetime, timezone
from typing import Optional

import aiosqlite
from fastapi import APIRouter, HTTPException, Query

from api.config import DB_PATH
from api.schemas import (
    LabelRequest, LabelResponse,
    SwingHistoryEntry, SwingsListResponse,
    SwingResultOut,
)

router = APIRouter(prefix="/swings", tags=["swings"])


def _row_to_history(row) -> SwingHistoryEntry:
    features = json.loads(row["features_json"] or "{}")
    return SwingHistoryEntry(
        id=row["id"],
        session_id=row["session_id"],
        swing_index=row["swing_index"],
        captured_at=row["captured_at"],
        verdict=row["verdict"],
        label=row["user_label"] if row["user_label"] is not None else row["label"],
        user_label=row["user_label"],
        confidence=row["confidence"],
        acc_mean=features.get("acc_mean"),
        omega_mean=features.get("omega_mean"),
        tempo_ratio=features.get("tempo_ratio"),
    )


@router.get("", response_model=SwingsListResponse)
async def list_swings(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    label: Optional[int] = Query(None),
):
    offset = (page - 1) * page_size
    where = "WHERE label = ?" if label is not None else ""
    params_count = [label] if label is not None else []
    params_query = params_count + [page_size, offset]

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            f"SELECT COUNT(*) FROM swing_results {where}", params_count
        ) as cur:
            total = (await cur.fetchone())[0]

        async with db.execute(
            f"SELECT * FROM swing_results {where} ORDER BY captured_at DESC LIMIT ? OFFSET ?",
            params_query,
        ) as cur:
            rows = await cur.fetchall()

    return SwingsListResponse(
        total=total,
        page=page,
        page_size=page_size,
        swings=[_row_to_history(r) for r in rows],
    )


@router.get("/{swing_id}", response_model=SwingResultOut)
async def get_swing(swing_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM swing_results WHERE id = ?", (swing_id,)
        ) as cur:
            row = await cur.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Swing not found")

    return SwingResultOut(
        id=row["id"],
        swing_index=row["swing_index"],
        verdict=row["verdict"],
        label=row["user_label"] if row["user_label"] is not None else row["label"],
        confidence=row["confidence"],
        features=json.loads(row["features_json"] or "{}"),
        feature_importances=json.loads(row["importances_json"] or "[]"),
        phase_timing=json.loads(row["phase_timing_json"] or "{}"),
    )


@router.post("/{swing_id}/label", response_model=LabelResponse)
async def label_swing(swing_id: str, req: LabelRequest):
    if req.label not in (0, 1):
        raise HTTPException(status_code=422, detail="label must be 0 or 1")

    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        result = await db.execute(
            "UPDATE swing_results SET user_label = ? WHERE id = ?",
            (req.label, swing_id),
        )
        await db.commit()
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Swing not found")

    return LabelResponse(id=swing_id, label=req.label, updated_at=now)
