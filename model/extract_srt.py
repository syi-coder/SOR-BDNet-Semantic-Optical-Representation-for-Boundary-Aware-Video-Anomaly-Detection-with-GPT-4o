import os
import json
import re

srt_dir = "/home/sun-y14/ucf-crimes/Anomaly-Videos-Part-1/caption/Assault038_x264/"
output_json_path = "/home/sun-y14/ucf-crimes/Anomaly-Videos-Part-1/caption/json/extracted_subtitles.json"

srt_files = sorted([f for f in os.listdir(srt_dir) if f.endswith(".srt")])

subtitles_dict = {}

for srt_file in srt_files:
    file_path = os.path.join(srt_dir, srt_file)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        match = re.search(r"Video Subtitle Generation:\s*\n(.*)", content, re.DOTALL)
        if match:
            subtitle_text = match.group(1).strip()
            file_num = int(srt_file.split(".")[0])
            subtitles_dict[str(file_num)] = subtitle_text

subtitles_dict = dict(sorted(subtitles_dict.items(), key=lambda x: int(x[0])))

os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(subtitles_dict, f, indent=4, ensure_ascii=False)

print(f"Extraction complete. Results saved to {output_json_path}")
