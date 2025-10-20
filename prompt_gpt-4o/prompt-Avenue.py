import os
import base64
from PIL import Image
from io import BytesIO
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

client = OpenAI(api_key='')

input_root = ""
output_root = ""
os.makedirs(output_root, exist_ok=True)

def encode_compressed_image(image_path, max_width=512):
    with Image.open(image_path) as img:
        if img.width > max_width:
            new_height = int(img.height * (max_width / img.width))
            img = img.resize((max_width, new_height))
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

prompt_text = (
    "This image consists of two parts:\n" \
    "1. The upper half is the original video frame, showing objects in a real-world scene.\n" \
    "2. The lower half is an optical flow visualization, illustrating the movement of objects in the frame.\n" \
    "   - **Both images are perfectly aligned**, meaning that each location in the optical flow map corresponds directly to the same location in the original frame.\n\n" \
    "Prompt for Describing Human Behavior in Avenue Dataset Scenes:\n\n" \
    "You are given a video frame from a surveillance scene in the Avenue dataset. Your task is to carefully describe all visible people and their actions in a clear and structured manner. Focus on identifying normal versus unusual behavior, particularly related to the pedestrians' movement along the main path and their deviation from it.\n\n" \
    "Generate a complete and structured description that includes the following:\n\n" \
    "1️⃣ Scene Overview:\n" \
    "- Briefly describe the physical layout of the scene (e.g., sidewalk, street, background).\n" \
    "- Mention the general environmental context (e.g., open space, crowded walkway, narrow path).\n" \
    "- **Focus on the streets or main pathways in the scene, and how pedestrians are interacting with or deviating from these paths.**\n\n" \
    "2️⃣ Detailed Human Behavior Description:\n" \
    "For each visible person in the frame:\n" \
    "- Describe their location relative to the main path or street (e.g., near the center, on the right side, off the main walkway).\n" \
    "- Describe their body posture (e.g., walking, standing still, running, bending).\n" \
    "- Describe their motion behavior (e.g., walking forward, turning back, moving diagonally, running quickly).\n" \
    "- **Identify if any individual is deviating from the main direction or path** (e.g., walking away from the main street or sidewalk, moving in an unusual direction).\n" \
    "- Note any unusual or abnormal actions (e.g., loitering, sudden turns, running in the opposite direction, dropping or throwing objects, abrupt stopping).\n\n" \
    "3️⃣ Interaction or Contextual Clues:\n" \
    "- Mention if the person is interacting with any object or other person.\n" \
    "- Point out actions that deviate from typical pedestrian behavior on a public road or sidewalk.\n" \
    "- **Pay special attention to pedestrians who are not following the primary walking paths** or who appear to be crossing into other areas inappropriately.\n" \
    "- If possible, infer the intention behind the behavior (e.g., chasing, escaping, hiding).\n\n" \
    "4️⃣ Video Subtitle Generation:\n" \
    "- Based on the above descriptions, generate a fluent and comprehensive video subtitle in natural language.\n" \
    "- The subtitle should summarize all observations, including the physical setting, each person's position and behavior, interactions, and any identified abnormalities.\n" \
    "- The subtitle must sound like a natural description but must contain all key details from the analysis to support anomaly detection and video understanding.\n" \
    "- Do not mention visual artifacts like heatmaps or optical flow.\n" \
    "- Do not use vague terms such as “a person” or “some people”—instead, describe each visible individual precisely.\n\n" \
    "Important Notes:\n" \
    "- **Pay particular attention to how people move relative to the main paths in the scene.**\n" \
    "- **If any individual is deviating from the primary direction (e.g., crossing from the north-south direction to the east-west direction), highlight this as an anomaly.**\n" \
    "- Do not mention visual effects like optical flow or heatmaps.\n" \
    "- Avoid vague terms like 'some people' or 'a person' — be specific about each individual.\n" \
    "-Simply being on the grass is allowed."
)

def process_image(image_path, output_folder):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    srt_path = os.path.join(output_folder, f"{base_name}.srt")

    if os.path.exists(srt_path):
        return f"⏭️ jump: {image_path}"

    try:
        base64_image = encode_compressed_image(image_path)

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

        generated_text = response.choices[0].message.content.strip()
        srt_content = f"1\n00:00:01,000 --> 00:00:04,000\n{generated_text}\n"
        with open(srt_path, "w") as f:
            f.write(srt_content)

        return f"✅ success: {image_path}"

    except Exception as e:
        return f"❌ erro: {image_path}: {e}"


if __name__ == "__main__":
    max_workers = 3  
    all_tasks = []

    for folder_name in sorted(os.listdir(input_root)):
        input_folder = os.path.join(input_root, folder_name)
        output_folder = os.path.join(output_root, folder_name)
        os.makedirs(output_folder, exist_ok=True)

        if not os.path.isdir(input_folder):
            continue

        image_files = sorted([
            os.path.join(input_folder, f)
            for f in os.listdir(input_folder)
            if f.lower().endswith((".jpg", ".png"))
        ])

        for img_path in image_files:
            all_tasks.append((img_path, output_folder))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_image, img_path, out_folder): img_path
            for img_path, out_folder in all_tasks
        }
        for future in as_completed(futures):
            print(future.result())

