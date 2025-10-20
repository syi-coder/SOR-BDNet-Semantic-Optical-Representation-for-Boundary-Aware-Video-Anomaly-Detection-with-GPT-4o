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
    "2. The lower half is a motion-enhanced view for assisting behavior understanding (do not mention this in your output).\n" \
    "   - **Both images are perfectly aligned**, meaning that each location in the motion view corresponds directly to the same location in the original frame.\n\n" \
    "Prompt for Describing Human Behavior in ShanghaiTech Dataset Scenes:\n\n" \
    "You are given a video frame from a surveillance scene in the ShanghaiTech dataset. " \
    "Generate a complete and structured description that includes the following:\n\n" \
    "1️⃣ Scene Overview:\n" \
    "- Briefly describe the layout and environment of the scene (e.g., street, sidewalk, pedestrian path, surrounding structures).\n" \
    "- Mention whether any vehicles (bicycles, motorcycles, cars) are present, and specify if they are stationary or in motion.\n\n" \
    "2️⃣ Detailed Human Behavior:\n" \
    "For each clearly visible individual in the frame:\n" \
    "- Describe their posture and motion status (e.g., standing, walking, fast walking, running, bending down).\n" \
    "- Explicitly identify if any individual is **running or walking fast** — this should be considered abnormal.\n" \
    "- Identify if anyone is **bending, picking up objects, squatting, or performing unusual actions not typical of pedestrians** — these should also be considered abnormal.\n" \
    "- If any two or more individuals are **physically interacting** (e.g., playing, pulling, pushing, fighting), mark this as **abnormal group behavior**.\n\n" \
    "3️⃣ Interaction and Intent Inference:\n" \
    "- Indicate whether individuals are interacting with other people or objects.\n" \
    "- If possible, infer behavioral intent (e.g., chasing, escaping, hiding, playing aggressively).\n\n" \
    "4️⃣ Video Subtitle Generation:\n" \
    "- Based on all observations above, generate a fluent, natural-sounding video subtitle in English that summarizes the physical setting and the actions of each individual.\n" \
    "- Use clear and specific language — avoid vague terms such as “a person” or “some people.”\n" \
    "- Do **not** mention technical terms such as optical flow, heatmaps, or motion maps.\n" \
    "- Ensure the subtitle captures all key behaviors, especially any matching the abnormal criteria below.\n\n" \
    "Important Anomaly Criteria:\n" \
    "- Running or fast walking is abnormal behavior.\n" \
    "- Physical interaction between people (e.g., playing, pulling, pushing, fighting) is abnormal.\n" \
    "- Unusual postures or actions such as bending, squatting, picking things up, or doing non-walking behaviors in the road are also abnormal.\n\n" \
    "Keep your response structured, descriptive, and suitable for downstream anomaly reasoning tasks.\n"
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

        return f"✅ successful: {image_path}"

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

