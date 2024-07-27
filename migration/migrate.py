from pydantic import BaseModel
from pymongo import MongoClient
from bson import ObjectId
import json

# connect to mongodb 
client = MongoClient('mongodb://localhost:27017')
db = client['test']
collection = db['images']

# function to migrate data
def migrate():
    # go to global_index2kf_path.json to see the data
    # the data should look like this
    # {
    #     "0": "Root\\L01\\V001\\0000000.jpg",
    #     "1": "Root\\L01\\V001\\0000028.jpg",
    # }
    # push each of them to mongodb, using Data model
    with open('global_index2kf_path.json', 'r') as f:
        # transform the data to a list of Data model
        # then push it to mongodb using push_many
        data = [Data(name=name, data=data).model_dump() for name, data in json.load(f).items()]
        collection.insert_many(data)


if __name__ == '__main__':
    migrate()