from functools import partial
from app.repositories.query import QueryRepository
from app.models.query_text_image import TextImage
from app.controllers.query import QueryController


class Factory:
    """
    This is the factory container that will instantiate all the controllers and
    repositories which can be accessed by the rest of the application.
    """
    # Repositories
    def query_repository(self):
        # Assuming TextImage is the collection you want to pass
        return QueryRepository(collection=TextImage)
    
    def get_query_controller(self):
        return QueryController(
            query_repository=self.query_repository()
        )
