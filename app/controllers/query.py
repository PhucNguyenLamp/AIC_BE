
from app.common.controller.base import BaseController
from app.models.text import Text
from app.repositories.query import QueryRepository
from app.models.text import Text
from app.repositories.query import QueryRepository


class QueryController(BaseController[Text]):
    def __init__(self, query_repository: QueryRepository):
        super().__init__(model=Text, repository=query_repository)
        self.query_repository = query_repository

    async def get_indexes_by_text(self, text: str) -> Text:
        # get all method in repository
        return await self.query_repository.get_record_by_index(text)