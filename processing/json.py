import os
import json

input_root = ""
output_root = ""
os.makedirs(output_root, exist_ok=True)

def extract_video_subtitle_generation(srt_path):
    with open(srt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    capture = False
    collected_lines = []
    for line in lines:
        if line.strip().startswith("4️⃣ Video Subtitle Generation:"):
            capture = True
            collected_lines.append(line.strip().replace("4️⃣ Video Subtitle Generation:", "").strip())
        elif capture:
            if line.strip() == "":
                break  
            collected_lines.append(line.strip())

    return " ".join(collected_lines) if collected_lines else None

for folder in sorted(os.listdir(input_root)):
    folder_path = os.path.join(input_root, folder)
    if not os.path.isdir(folder_path):
        continue

    result = {}
    for fname in sorted(os.listdir(folder_path)):
        if fname.endswith(".srt"):
            srt_path = os.path.join(folder_path, fname)
            content = extract_video_subtitle_generation(srt_path)
            if content:
                key = str(int(os.path.splitext(fname)[0]))  
                result[key] = content

    if result:
        json_path = os.path.join(output_root, f"{folder}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        print(f"✅ ：{json_path}")
    else:
        print(f"⚠ jump：{folder}（useless）")

print("🎉 ")
