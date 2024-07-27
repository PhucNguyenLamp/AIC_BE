from typing import List
from app.common.controller import BaseController
from app.common.exceptions import NotFoundException
from app.models import Text
from app.repositories import QueryRepository
from app.schemas.requests import SearchBodyRequest

class QueryService(BaseController[Text]):
    def __init__(self, query_repository: QueryRepository):
        super().__init__(model=Text, repository=query_repository)
        self.query_repository = query_repository

    async def get_keyframe_by_index(self, index: int) -> Text:
        result = await self.query_repository.get_keyframe_by_index(index)

        if not result:
            raise NotFoundException("No keyframe found")

        return result

    async def search_get_keyframe_index(
        self, body: List[SearchBodyRequest]
    ) -> List[int]:
        # do something
        print(f"search_get_keyframe_index ${len(body)}")

        return [1, 2]
