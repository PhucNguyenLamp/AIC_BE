from typing import List
from app.common.factory import Factory
from fastapi import APIRouter, Body, Depends, Query
from app.models import Text
from app.schemas.extras import Response
from app.schemas.requests import SearchBodyRequest
from app.services import QueryService
from app.controllers import QueryController

query_router = APIRouter()

def get_query_controller(
    query_service: QueryService = Depends(Factory().get_query_service),
):
    return QueryController(query_service)


@query_router.post(
    "/search",
    response_model=Response[List[int]],
    summary="Search keyframes",
    tags=["Query"],
    response_description="List of keyframe indexes",
    status_code=200,
    description="Predict to get top k similar keyframes",
)
async def search(
    query_service: QueryService = Depends(Factory().get_query_service),
    request_body: List[SearchBodyRequest] = Body(),
) -> Response[List[int]]:
    query_controller = get_query_controller(query_service)

    query = await query_controller.search_get_keyframe_index(request_body)

    return Response[List[int]](data=query)


@query_router.get(
    "/",
    tags=["Query"],
    response_model=Response[Text],
    status_code=200,
    description="Query one record by index",
)
async def query_one_by_index(
    query_service: QueryService = Depends(Factory().get_query_service),
    index: int = Query(example=1, description="Index of the keyframe"),
) -> Response[Text]:
    query_controller = get_query_controller(query_service)

    query = await query_controller.get_keyframe_by_index(index=index)

    return Response[Text](data=query)
