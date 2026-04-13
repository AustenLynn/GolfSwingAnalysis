import sys
from fastapi import APIRouter, HTTPException
from api.config import SCRIPTS_DIR, FEATURES_CSV
from api.schemas import ModelInfoResponse, RetrainResponse
from api.services.model_service import ModelService

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/info", response_model=ModelInfoResponse)
async def model_info():
    obj = ModelService.get()
    clf = obj["model"].named_steps.get("clf") if obj["type"] == "sklearn" else None
    model_class = type(clf).__name__ if clf else obj["type"]

    # Count training samples from swing_features.csv
    n_good, n_bad = 0, 0
    if FEATURES_CSV.exists():
        import pandas as pd
        df = pd.read_csv(FEATURES_CSV)
        n_good = int((df["label"] == 1).sum())
        n_bad = int((df["label"] == 0).sum())

    return ModelInfoResponse(
        model_type=obj["type"],
        model_class=model_class,
        features=obj["features"],
        loocv_accuracy=None,  # stored in pkl metadata after retrain
        n_training_samples=n_good + n_bad,
        n_good=n_good,
        n_bad=n_bad,
    )


@router.post("/retrain", response_model=RetrainResponse)
async def retrain():
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        import train_classifier
        import importlib
        importlib.reload(train_classifier)
        train_classifier.main()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrain failed: {exc}") from exc

    ModelService.reload()

    n_good, n_bad = 0, 0
    if FEATURES_CSV.exists():
        import pandas as pd
        df = pd.read_csv(FEATURES_CSV)
        n_good = int((df["label"] == 1).sum())
        n_bad = int((df["label"] == 0).sum())

    obj = ModelService.get()
    clf = obj["model"].named_steps.get("clf") if obj["type"] == "sklearn" else None

    return RetrainResponse(
        status="ok",
        loocv_accuracy=0.0,   # train_classifier prints it; we don't capture it here
        n_good=n_good,
        n_bad=n_bad,
        model_type=type(clf).__name__ if clf else obj["type"],
    )
