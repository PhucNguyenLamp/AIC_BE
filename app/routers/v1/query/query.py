from app.common.factory import Factory
from fastapi import APIRouter, Depends, Request
from app.services.query import QueryService
from app.controllers.query import QueryController

query_router = APIRouter()

def get_query_controller(query_service: QueryService = Depends(Factory().get_query_service)):
    return QueryController(query_service)

@query_router.get("/", response_model=list[dict])
async def query_by_text(
    request: Request,
    query_service: QueryService = Depends(Factory().get_query_service)
) -> list[dict]:
    query_controller = get_query_controller(query_service)

    if request.query_params:
        print(request.query_params)

    query = await query_controller.get_keyframe_by_index(index = request.query_params["index"])

    return [{"message": "Okay", "data": query}]
