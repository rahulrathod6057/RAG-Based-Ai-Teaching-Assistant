# Converting the video to mp3 

import os
import subprocess

files = os.listdir("videos")


for file in files:
    file_name = file.split("+")[0]
    tutorial_number = file.split(".")[0].split("#")[1]
    print(tutorial_number,file_name)
    subprocess.run(["ffmpeg", "-i", f"videos/{file}", f"audios/{tutorial_number}_{file_name}.mp3"])

