import os
import math
import json

n = 5

for filename in os.listdir("jsons"):
    if filename.endswith(".json"):
        file_path = os.path.join("jsons", filename)
        with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                new_chunks = []
                num_chunks = len(data["chunks"])
                num_groups = math.ceil(num_chunks / n)

                for i in range(num_groups):
                    start_idx = i * n
                    end_idx = min((i + 1) * n, num_chunks)

                    chunks_group = data["chunks"][start_idx:end_idx]

                    new_chunks.append({
                        "numbers": data["chunks"][0]["numbers"],
                        "title": data["chunks"][0]["title"],
                        "start": chunks_group[0]["start"],
                        "end": chunks_group[-1]["end"],
                        "text": " ".join([chunk["text"] for chunk in chunks_group])
                    })


                #Save file without double .json
                os.makedirs("newJsons", exist_ok=True)
                with open(os.path.join("newJsons", filename), "w", encoding="utf-8") as json_file:
                     json.dump({"chunks": new_chunks, "text": data["text"]}, json_file, indent=4)
                     