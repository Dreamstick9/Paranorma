from datasets import load_dataset
import pandas as pd
import voyageai
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

my_api_key = os.getenv("VOYAGE_API_KEY")
my_db_uri = os.getenv("MONGO_URI")

# downloaded the data from hf
url = "https://huggingface.co/datasets/lukeslp/strange-places-mysterious-phenomena/resolve/main/strange_places_v5.2.json"
df = pd.read_json(url)
df = df.head(500)

# Connected with the embedding model
vo = voyageai.Client(api_key=my_api_key)

# Function for string input to vector ouput
def get_embedding(text):
    text = str(text)
    if text == '':
        return []
    text_embeddings = vo.embed([text], model="voyage-4-large", input_type="document").embeddings

    return text_embeddings[0]

# applying the function to all the rows
target = df["description"]

df["embedding"] = target.apply(get_embedding)

# storing the embeddings in the mogo db database
uri = my_db_uri
client = MongoClient(uri)

database = client["locations"]
collection = database["strange_loactions"]

collection.delete_many({})

data_to_insert = df.to_dict(orient ="records")

collection.insert_many(data_to_insert)

print(len(data_to_insert))