from openai import OpenAI
import json
import os
import re


client = OpenAI(api_key='')


train_dir = ""
test_dir = ""
output_dir = ""
os.makedirs(output_dir, exist_ok=True)

train_descriptions = []
train_files = sorted([os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(".json")])
for file in train_files:
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        train_descriptions.extend(data.values())
train_descriptions_sample = train_descriptions[:50]


train_summary_prompt = f"""
你是一名 AI 语言分析专家，请分析以下文本的主要内容：
{train_descriptions_sample}

请总结：
1. 这些描述的主要场景是什么？
2. 是否涉及交通工具？（如果有，请列出）
3. 主要人物行为模式是什么？
4. 其他值得注意的共同特征？

请以 **JSON 格式** 返回：
{{
    "场景": "...",
    "交通工具": "...",
    "行为模式": "...",
    "其他特征": "..."
}}
"""
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": train_summary_prompt}],
    temperature=0.7
)
train_summary = response.choices[0].message.content
print("✅ ")

# 遍历所有测试文件
test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".json")])
for test_file in test_files:
    anomalies = []
    test_path = os.path.join(test_dir, test_file)

    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    print(f"\n📂 正在处理测试文件：{test_file}（共 {len(test_data)} 条描述）")

    for key, test_desc in test_data.items():
        print(f"🔍 分析编号 {key}: {test_desc[:50]}...")

        # 更严格的判断标准
        test_analysis_prompt = f"""
        训练集的总结如下：
        {train_summary}

        请你作为行为分析专家，判断以下文本是否与训练集的模式一致。
        注意：我们采用**严格匹配**标准：
        - 只要文本中出现了**训练集中未出现过的行为、动作或移动方式**，即判定为异常。
        - 特别注意：**奔跑、快走、弯腰、捡东西、非正常方向移动、打闹或互动**等行为，若在训练集中未出现，一律视为异常。
        - 我们只关注是否符合训练集中行为习惯，不考虑是否属于合理行为。
        -允许描述一条安静的街道，没有行人或车辆的存在，这是正常的；允许行人使用人行横道过马路。
        现在给你一个新的文本：
        "{test_desc}"

        请判断：
        1. 这个文本是否严格符合训练集的行为模式？（是 / 不是）
        2. 如果不是，请详细说明偏离的具体原因（如出现了未见过的动作、互动形式、人物行为模式等）
        3. **如果不匹配，请将“是否异常”设为 "yes"**

        请用以下 JSON 格式回答：
        {{
            "匹配训练集": "yes" 或 "no",
            "偏离原因": "...",
            "是否异常": "yes" 或 "no"
        }}
        """

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": test_analysis_prompt}],
            temperature=0.7
        ).choices[0].message.content

        def extract_json(text):
            match = re.search(r'\{.*\}', text, re.DOTALL)
            return match.group(0) if match else None

        try:
            extracted_json = extract_json(response)
            if extracted_json:
                analysis_result = json.loads(extracted_json)
                if analysis_result.get("是否异常") == "yes":
                    anomalies.append({
                        "编号": key,
                        "测试文本": test_desc,
                        "分析结果": analysis_result
                    })
            else:
                print(f"⚠️ 无法提取 JSON，跳过编号 {key}")
        except Exception as e:
            print(f"⚠️ JSON 解析失败：{e}")

    # 保存结果
    output_path = os.path.join(output_dir, test_file.replace(".json", "_result.json"))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(anomalies, f, indent=4, ensure_ascii=False)
    print(f"✅ 保存结果至：{output_path}（异常条数：{len(anomalies)}）")

print("\n🎉 全部测试文件处理完毕！")