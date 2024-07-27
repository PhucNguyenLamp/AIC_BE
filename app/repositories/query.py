from typing import Any, List
from app.common.repository.base import BaseRepository
from app.models.text import Text


class QueryRepository(BaseRepository[Text]):
    """
    Query repository provides all the database operations for the Query model.
    """

    async def get_record_by_index(self, index: Any) -> List[Text]:
        """
        Get all record by index.

        :param text: index
        :return: A list of tasks.
        """
        return await self.get_all()
