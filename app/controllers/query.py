
from app.common.controller.base import BaseController
from app.models.text import Text
from app.repositories.query import QueryRepository
from app.models.text import Text
from app.repositories.query import QueryRepository
from app.services.query import QueryService

class QueryController():
    def __init__(self, query_serivce: QueryService):
        self.query_serivce = query_serivce

    async def get_keyframe_by_index(self, index: int) -> Text:
        # get all method in repository
        return await self.query_serivce.get_keyframe_by_index(index)