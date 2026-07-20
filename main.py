from datasets import load_dataset
import pandas as pd
import voyageai
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

my_api_key = os.getenv("VOYAGE_API_KEY")
my_db_uri = os.getenv("MONGO_URI")

# # downloaded the data from hf
url = "https://huggingface.co/datasets/lukeslp/strange-places-mysterious-phenomena/resolve/main/strange_places_v5.2.json"
df = pd.read_json(url)
df = df.sample(n=500, random_state=42)

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

def vector_Search(query):
    query_embedding = vo.embed([query], model="voyage-4-large", input_type="query").embeddings[0]

    if query_embedding is None:
        return "Embedding generation failed bruh try that again"
    
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": 150,
                "limit": 5,
            }
        },
        {
            "$project": {
                "_id": 0,
                "embedding": 0,
                "score": {
                    "$meta": "vectorSearchScore"
                },
            }
        },
    ]
    results = collection.aggregate(pipeline)
    return list(results)



bclient = OpenAI(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url="https://opencode.ai/zen/v1" 
)

def handle_user_query(query):
    retrieval = vector_Search(query)
    search_result = ""
    for row in retrieval:
        search_result+= f"Location: {row.get('name', 'Unknown')}\n"
        search_result+= f"Description: {row.get('description', 'No description found.')}\n"
        search_result+= "-" * 20 + "\n"

    response = bclient.chat.completions.create(
        model="hy3-free",
        max_tokens=1024,

        messages=[
            {
                "role": "system",
                "content": "You are a helpful expert on mysterious places and phenomena. Answer the user's question using ONLY the information provided in the Context below. If the context doesn't contain the answer, politely say 'I dont have enough information in my database to answer that.'"
            },
            {
                "role": "user",
                "content": "Answer this user query: " + query + " with the following context: " + search_result,
            }
        ],
    )

    return response.choices[0].message.content



print(handle_user_query("Are there any reports of a Bigfoot or large bipedal creature sighting?"))

