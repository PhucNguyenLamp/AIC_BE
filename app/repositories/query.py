from typing import Any, List
from app.common.repository.base import BaseRepository
from app.models.query_text_image import TextImage


class QueryRepository(BaseRepository[TextImage]):
    """
    Query repository provides all the database operations for the Query model.
    """
    async def get_record_by_index(self, index: Any) -> List[TextImage]:
        """
        Get all record by index.

        :param text: index
        :return: A list of tasks.
        """
        return await self.get_all()
