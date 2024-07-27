from typing import List
from app.models.text import Text
from app.models.text import Text
from app.schemas.requests.query import SearchBodyRequest
from app.services.query import QueryService

class QueryController():
    def __init__(self, query_serivce: QueryService):
        self.query_serivce = query_serivce

    async def get_keyframe_by_index(self, index: int) -> Text:
        # get all method in repository
        return await self.query_serivce.get_keyframe_by_index(index)

    async def search_get_keyframe_index(self, body: List[SearchBodyRequest]):
        return await self.query_serivce.search_get_keyframe_index(body)
