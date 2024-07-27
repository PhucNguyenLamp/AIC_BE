from functools import partial
from app.repositories.query import QueryRepository
from app.models.text import Text
from app.controllers.query import QueryController


class Factory:
    """
    This is the factory container that will instantiate all the controllers and
    repositories which can be accessed by the rest of the application.
    """
    # Repositories
    def query_repository(self):
        # Assuming Text is the collection you want to pass
        return QueryRepository(collection=Text)
    
    def get_query_controller(self):
        return QueryController(
            query_repository=self.query_repository()
        )
