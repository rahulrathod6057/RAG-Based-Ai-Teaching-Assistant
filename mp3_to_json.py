import whisper
import json
import os 

model = whisper.load_model("large-v2")

audios = os.listdir("audios")

for audio in audios:
    print(audio)
    if "_" in audio:
        numbers = audio.split("_")[0]
        title = audio.split("_")[1][:-4]
        # print(numbers,title)
        result = model.transcribe(
                                f"audios/{audio}",
                                language="hi",
                                task="translate",
                                word_timestamps=False
                                )

        chunks = []
        for segment in result["segments"]:
            chunks.append({
                "numbers": numbers,
                "title": title,
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            })

        chunks_with_metadata = {"chunks": chunks, "text": result["text"]}

        with open(f"jsons/{audio}.json", "w") as f:
            json.dump(chunks_with_metadata, f)

        