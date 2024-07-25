from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from app.routers.v1 import v1_router
from app.config.config import settings
from app.models.query_text_image import TextImage
from beanie import init_beanie
from contextlib import asynccontextmanager
from motor.motor_asyncio import AsyncIOMotorClient

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Starting application", settings)
    # Setup mongoDB
    app.client = AsyncIOMotorClient(
        settings.MONGO_HOST,
        settings.MONGO_PORT,
        # username=settings.MONGO_USER,
        # password=settings.MONGO_PASSWORD,
    )
    await init_beanie(database=app.client[settings.MONGO_DB], document_models=[TextImage])

    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(v1_router, prefix=settings.API_V1_STR)
