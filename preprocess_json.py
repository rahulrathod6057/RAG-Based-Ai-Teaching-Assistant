import requests
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib


def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()["embeddings"]
    return embedding


jsons = os.listdir("newJsons")
my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"newJsons/{json_file}") as f:
        content = json.load(f)

    print(f"Creating Embeddings for {json_file}")

    chunks = content["chunks"]

    for start in range(0, len(chunks), 20):
        batch = chunks[start:start + 20]

        embeddings = create_embedding([c["text"] for c in batch])

        for i, chunk in enumerate(batch):
            chunk["chunk_id"] = chunk_id
            chunk["embedding"] = embeddings[i]
            chunk_id += 1
            my_dicts.append(chunk)
            

df = pd.DataFrame.from_records(my_dicts)
#save the dataframe 
joblib.dump(df, "embeddings.joblib")

