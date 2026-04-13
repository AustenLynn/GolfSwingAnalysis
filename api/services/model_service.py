import pickle
from api.config import MODEL_PATH


class ModelService:
    _model_obj: dict | None = None

    @classmethod
    def load(cls) -> None:
        with open(MODEL_PATH, "rb") as f:
            cls._model_obj = pickle.load(f)
        print(f"[ModelService] Loaded model: {MODEL_PATH}")

    @classmethod
    def get(cls) -> dict:
        if cls._model_obj is None:
            cls.load()
        return cls._model_obj

    @classmethod
    def reload(cls) -> None:
        cls._model_obj = None
        cls.load()

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._model_obj is not None
