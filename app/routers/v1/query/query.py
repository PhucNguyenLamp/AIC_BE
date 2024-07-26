from app.common.factory import Factory
from fastapi import APIRouter, Depends, Request
from app.controllers.query import QueryController

query_router = APIRouter()

@query_router.get("/", response_model=list[dict])
async def query_by_text(
    request: Request,
    query_controller: QueryController = Depends(Factory().get_query_controller)
) -> list[dict]:
    query = await query_controller.get_indexes_by_text('123')

    return [{"message": "Okay", "data": query}]
