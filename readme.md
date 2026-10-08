# How to Use This RAG-Based AI Teaching Assistant on Your Own Data

## Step 1 - Collect Your Videos

Move all your video files into the `videos` folder.

## Step 2 - Convert Videos to MP3

Convert all the video files to MP3 by running `video_to_mp3.py`.

## Step 3 - Convert MP3 to JSON

Convert all the MP3 files to JSON by running `mp3_to_json.py`.

## Step 4 - Convert JSON Files to Vectors

Use `preprocess_json.py` to process the JSON files, create embeddings, convert the data into a DataFrame, and save it as `embeddings.joblib`.

## Step 5 - Prompt Generation and Feeding to the LLM

Load the `embeddings.joblib` file into memory. Then create a relevant prompt based on the user's query and feed it to the LLM.


# Tip

For better output, merge the chunks using the file `merge_chunks.py`.

## Optional

`process_incoming.py` can be used to test the RAG pipeline directly from the terminal without using the Streamlit interface.

For the Streamlit dashboard, run:

```bash
py -m streamlit run app.py