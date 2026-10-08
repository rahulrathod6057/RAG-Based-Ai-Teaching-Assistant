import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import requests
import json


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Teaching Assistant",
    page_icon="🎓",
    layout="wide"
)


# --------------------------------------------------
# LOAD EMBEDDINGS
# --------------------------------------------------

@st.cache_resource
def load_data():
    return joblib.load("embeddings.joblib")


df = load_data()


# --------------------------------------------------
# EMBEDDING FUNCTION
# --------------------------------------------------

def create_embedding(text_list):

    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )

    r.raise_for_status()

    embedding = r.json()["embeddings"]

    return embedding


# --------------------------------------------------
# STREAMING LLAMA FUNCTION
# --------------------------------------------------

def inference(prompt):

    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": True
        },
        stream=True
    )

    r.raise_for_status()

    for line in r.iter_lines():

        if line:

            data = json.loads(line.decode("utf-8"))

            if "response" in data:
                yield data["response"]

            if data.get("done"):
                break


# --------------------------------------------------
# RAG SEARCH
# --------------------------------------------------

def search_chunks(incoming_query):

    # Create embedding for question
    question_embedding = create_embedding(
        [incoming_query]
    )[0]

    # Compare with stored embeddings
    similarities = cosine_similarity(
        np.vstack(df["embedding"].values),
        [question_embedding]
    ).flatten()

    # Get top 5 results
    top_result = 5

    max_indices = similarities.argsort()[::-1][:top_result]

    new_df = df.iloc[max_indices].copy()

    # Add similarity score
    new_df["similarity"] = similarities[max_indices]

    return new_df


# --------------------------------------------------
# CREATE PROMPT
# --------------------------------------------------

def create_prompt(incoming_query, new_df):

    prompt = f'''
I am teaching Python programming and also other Topics as well in this course. Here are video 
subtitle chunks containing video title, video number, start time in seconds, end time in 
seconds, and the text at that time :

{new_df[["title", "numbers", "start", "end", "text"]].to_json(orient="records")}

---------------------------------------------

"{incoming_query}"

User asked this question related to the video chunks, you have to answer in a human way
(dont mention the above format, its just for you)
where and how much content is taught in which video (in this video and at what timestamp)
and guide the user to go to that particular video.

If user asks unrelated question, tell him that you can only answer
questions related to the course.
'''

    return prompt


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("🎓 AI Teaching Assistant")

    st.write("### Course")
    st.write("Python Programming & Other Topics")

    st.divider()

    st.write("### Models")
    st.write("🧠 Embedding: `bge-m3`")
    st.write("🤖 LLM: `llama3.2`")

    st.divider()

    st.write("### Dataset")
    st.write(f"📚 Total Chunks: `{len(df)}`")

    st.divider()

    if st.button("Clear Chat"):

        st.session_state.messages = []

        st.rerun()


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🎓 AI Teaching Assistant")

st.caption(
    "Ask questions about the course and find the relevant video and timestamp."
)


# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

incoming_query = st.chat_input(
    "Ask something about the course..."
)


# --------------------------------------------------
# PROCESS QUESTION
# --------------------------------------------------

if incoming_query:

    # ----------------------------------------------
    # USER MESSAGE
    # ----------------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": incoming_query
    })

    with st.chat_message("user"):
        st.markdown(incoming_query)


    # ----------------------------------------------
    # ASSISTANT MESSAGE
    # ----------------------------------------------

    with st.chat_message("assistant"):

        try:

            # Find relevant chunks
            new_df = search_chunks(incoming_query)

            # Create prompt
            prompt = create_prompt(
                incoming_query,
                new_df
            )

            # --------------------------------------
            # STREAM RESPONSE
            # --------------------------------------

            response = st.write_stream(
                inference(prompt)
            )


            # --------------------------------------
            # SHOW SOURCES
            # --------------------------------------

            st.markdown("### 📚 Relevant Videos")


            for _, row in new_df.iterrows():

                start_time = int(row["start"])
                end_time = int(row["end"])

                minutes = start_time // 60
                seconds = start_time % 60

                timestamp = f"{minutes}:{seconds:02d}"


                with st.expander(
                    f"🎬 Video {row['numbers']} — "
                    f"{row['title']} — "
                    f"{timestamp}"
                ):

                    st.write(
                        f"**Timestamp:** "
                        f"{timestamp} "
                        f"({start_time}s - {end_time}s)"
                    )

                    st.write(
                        f"**Similarity:** "
                        f"{row['similarity']:.3f}"
                    )

                    st.write(
                        f"**Transcript:** "
                        f"{row['text']}"
                    )


            # --------------------------------------
            # SAVE ASSISTANT RESPONSE
            # --------------------------------------

            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })


        except Exception as e:

            error_message = f"Error: {e}"

            st.error(error_message)

            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message
            })