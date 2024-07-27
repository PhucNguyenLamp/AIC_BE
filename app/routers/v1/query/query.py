from app.common.factory import Factory
from fastapi import APIRouter, Depends, Query
from app.models.text import Text
from app.schemas.extras.common import Response
from app.services.query import QueryService
from app.controllers.query import QueryController

query_router = APIRouter()


def get_query_controller(
    query_service: QueryService = Depends(Factory().get_query_service),
):
    return QueryController(query_service)


@query_router.get(
    "/",
    response_model=Response,
    status_code=200,
    description="Query one record by index",
)
async def query_by_text(
    query_service: QueryService = Depends(Factory().get_query_service),
    index: int = Query(example=1, description="Index of the keyframe"),
) -> Response:
    query_controller = get_query_controller(query_service)

    query = await query_controller.get_keyframe_by_index(index=index)

    return Response[Text](data=query)
