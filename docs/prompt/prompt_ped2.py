import os
import base64
from openai import OpenAI

client = OpenAI(api_key='')

input_root = ""
output_root = ""
os.makedirs(output_root, exist_ok=True) 

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

prompt_text = (
    "This image was captured on a university campus.\n\n"
    "This image consists of two parts:\n"
    "1. The upper half is the original video frame, showing objects in a real-world scene.\n"
    "2. The lower half is an optical flow visualization, illustrating the movement of objects in the frame.\n"
    "   - **Both images are perfectly aligned, meaning that each location in the optical flow map corresponds directly to the same location in the original frame.**\n\n"
    "Generate four separate descriptions:\n\n"
    "### **1️⃣ Description of Objects in the Original Frame:**\n"
    "- List and describe all visible objects in the original scene.\n"
    "- Focus on their presence and characteristics **without mentioning movement.**\n\n"
    "### **2️⃣ Description of Motion Direction and Behavior:**\n"
    "- Describe how each object is moving based on its position and orientation.\n"
    "- **Explain the actions being performed by each moving object.**\n"
    "- Ensure that objects in the original frame are accurately described in their movements.\n"
    "- Indicate different movement patterns and interactions between objects.\n"
    "- If some objects are static (e.g., trees, benches), state that they are not moving.\n\n"
    "### **3️⃣ Description of Motion Speed:**\n"
    "- Describe which objects are moving faster or slower based on their appearance and relative positions.\n"
    "- **Do not mention direction here.**\n"
    "- Explain which elements appear to be in motion at varying speeds.\n"
    "- Identify objects that remain stationary.\n\n"
    "### **4️⃣ Subtitle for the Video:**\n"
    "- Summarize the key details from the three descriptions into a natural and detailed video subtitle.\n"
    "- The subtitle should clearly describe the scene, the movement of objects, and their speed differences.\n"
    "- Ensure that the subtitle is structured and complete, covering objects, motion direction, and speed effectively.\n"
    "- Do not mention optical flow or colors in the summary.\n"
)

for subdir in sorted(os.listdir(input_root)):
    subdir_path = os.path.join(input_root, subdir)
    if not os.path.isdir(subdir_path):
        continue

    output_subdir = os.path.join(output_root, subdir)
    os.makedirs(output_subdir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(subdir_path) if f.endswith(".png")])

    for image_file in image_files:
        image_path = os.path.join(subdir_path, image_file)
        base64_image = encode_image(image_path)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ],
        )

        generated_text = response.choices[0].message.content
        print(f"Processed: {subdir}/{image_file}")

        text_output_path = os.path.join(output_subdir, f"{os.path.splitext(image_file)[0]}_description.txt")
        with open(text_output_path, "w") as text_file:
            text_file.write(generated_text)

        subtitle_marker = "### **4️⃣ Subtitle for the Video"
        subtitle_start = generated_text.find(subtitle_marker)
        if subtitle_start != -1:
            subtitle_text = generated_text[subtitle_start:].strip()
        else:
            subtitle_text = "No subtitle generated."

        srt_content = f"1\n00:00:01,000 --> 00:00:04,000\n{subtitle_text}\n"
        srt_output_path = os.path.join(output_subdir, f"{os.path.splitext(image_file)[0]}.srt")
        with open(srt_output_path, "w") as srt_file:
            srt_file.write(srt_content)

        print(f"Saved: {text_output_path} and {srt_output_path}")

print("All images processed successfully!")
