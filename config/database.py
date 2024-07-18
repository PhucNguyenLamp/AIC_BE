from pymongo import MongoClient

uri = "mongodb://localhost:27017/"
client = MongoClient(uri)
db = client.images_db

collection_name = db["images_collection"]
