from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.database import init_db
from api.services.model_service import ModelService
from api.routers import analyze, swings, model, health

app = FastAPI(title="Golf Swing Analyzer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    await init_db()
    ModelService.load()


app.include_router(health.router)
app.include_router(analyze.router)
app.include_router(swings.router)
app.include_router(model.router)
