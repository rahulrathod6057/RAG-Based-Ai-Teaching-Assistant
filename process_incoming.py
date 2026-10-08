import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import requests

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()["embeddings"]
    return embedding

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.2",
        "prompt": prompt,
        "stream": True
    })

    response = r.json()
    print(response)
    return response
    

df = joblib.load("embeddings.joblib")

incoming_query = input("Ask About Qustion :")
question_embedding = create_embedding([incoming_query])[0] 

# Find similarity of qustion embedding with other emdeddings
# print(np.vstack(df['embedding'].values))
# print(np.vstack(df['embedding'].shape))

similarities = cosine_similarity(np.vstack(df['embedding'].values), [question_embedding]).flatten()
# print(similarities)
top_result = 5
max_indices = similarities.argsort()[::-1][0:top_result]  # Get the indices of the top 3 most similar chunks
# print(max_indices)
new_df = df.iloc[max_indices]
# print(new_df[["title", "numbers", "text"]])


prompt = f''' I am teaching Python programming and also other Topics as well in this course. Here are video 
subtitle chunks containing video title, video number, start time in seconds, end time in 
seconds, and the text at that time :

{new_df[["title", "numbers", "start", "end", "text"]].to_json(orient="records")}
---------------------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format,its just for you)
where and how much content is taught in which video (in this video and at what timestamp) and guide the user to go to 
that particular video. If user asks unrelated question, tell him that you can only answer
questions related to the course.
'''


with open ("prompt.txt", "w") as f:
    f.write(prompt)

response = inference(prompt)["response"]
print(response)

with open ("response.txt", "w") as f:
    f.write(response)
# for index, item in new_df.iterrows():
#     print(index, item["title"], item["numbers"], item["text"], item["start"], item["end"])

