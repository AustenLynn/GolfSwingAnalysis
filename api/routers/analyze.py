import json
import uuid
from datetime import datetime, timezone

import aiosqlite
from fastapi import APIRouter, HTTPException

from api.config import DB_PATH
from api.schemas import AnalyzeRequest, AnalyzeResponse, SwingResultOut
from api.services.model_service import ModelService
from api.services.pipeline_service import analyze_samples

router = APIRouter(tags=["analyze"])


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    model_obj = ModelService.get()
    samples = [s.model_dump() for s in req.samples]

    try:
        swing_results = analyze_samples(samples, model_obj)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc

    if not swing_results:
        return AnalyzeResponse(
            session_id=req.session_id,
            swings_detected=0,
            swings=[],
        )

    # Persist to DB
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR IGNORE INTO swing_sessions VALUES (?, ?, ?, ?)",
            (req.session_id, req.captured_at, len(samples), len(swing_results)),
        )
        swing_ids = []
        for r in swing_results:
            sid = str(uuid.uuid4())
            swing_ids.append(sid)
            await db.execute(
                """INSERT INTO swing_results
                   (id, session_id, swing_index, captured_at, verdict, label,
                    user_label, confidence, features_json, importances_json, phase_timing_json)
                   VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?)""",
                (
                    sid,
                    req.session_id,
                    r["swing_index"],
                    req.captured_at,
                    r["verdict"],
                    r["label"],
                    r["confidence"],
                    json.dumps(r["features"]),
                    json.dumps(r["feature_importances"]),
                    json.dumps(r["phase_timing"]),
                ),
            )
        await db.commit()

    out_swings = []
    for sid, r in zip(swing_ids, swing_results):
        out_swings.append(SwingResultOut(id=sid, **r))

    return AnalyzeResponse(
        session_id=req.session_id,
        swings_detected=len(out_swings),
        swings=out_swings,
    )
