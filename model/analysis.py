from openai import OpenAI
import json
import os
import re

client = OpenAI(api_key="")

train_dir = "/home/sun-y14/lavad-main/datasets/train_caption/"
train_files = sorted(
    [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(".json")]
)

train_descriptions = []
for file in train_files:
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        train_descriptions.extend(data.values())

train_samples = train_descriptions[:50]

train_summary_prompt = f"""
You are a language analysis expert. Please analyze the following text samples:
{train_samples}

Summarize the following:
1. What are the main scene patterns?
2. Are any transportation types mentioned? (List if present)
3. What are the main human behavior patterns?
4. Any other notable common features?

Return the result strictly in JSON format:
{{
    "scene": "...",
    "transportation": "...",
    "behaviors": "...",
    "other_features": "..."
}}
"""

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": train_summary_prompt}],
    temperature=0.7,
)
train_summary = response.choices[0].message.content

print("Training set summary:", train_summary)

test_file = "/home/sun-y14/lavad-main/datasets/linshi/004.json"
with open(test_file, "r", encoding="utf-8") as f:
    test_data = json.load(f)

print(f"Loaded {len(test_data)} test descriptions.")

anomalies = []

for key, test_desc in test_data.items():
    print(f"Processing {key}: {test_desc[:50]}...")

    test_prompt = f"""
The summary of the training set is as follows:
{train_summary}

Now analyze the following text:
"{test_desc}"

Please determine:
1. Does this text follow the patterns found in the training set? (yes / no)
2. If not, describe what deviates (e.g., new objects, unusual behavior, unexpected direction)
3. If the text does not match the training pattern, set "is_anomaly" to "yes".

Return strictly in JSON format:
{{
    "match_training": "yes" or "no",
    "deviation_reason": "...",
    "is_anomaly": "yes" or "no"
}}
"""

    test_result = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": test_prompt}],
        temperature=0.7,
    ).choices[0].message.content

    print("Result:", test_result)

    def extract_json(text):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return match.group(0) if match else None

    try:
        json_part = extract_json(test_result)
        if json_part:
            result = json.loads(json_part)
            if result.get("match_training") == "no":
                result["is_anomaly"] = "yes"
                anomalies.append(
                    {
                        "id": key,
                        "text": test_desc,
                        "analysis": result,
                    }
                )
        else:
            print(f"JSON not found for {key}")
            continue

    except json.JSONDecodeError:
        print(f"JSON decode failed for {key}")
        continue

if anomalies:
    output_file = "/home/sun-y14/lavad-main/datasets/linshi/004.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(anomalies, f, indent=4, ensure_ascii=False)
    print(f"Anomalies saved to {output_file}")
else:
    print("No anomalies detected.")
