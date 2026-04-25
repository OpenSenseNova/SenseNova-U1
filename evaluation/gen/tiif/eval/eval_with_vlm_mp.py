import argparse
import base64
import glob
import json
import math
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, wait

from openai import AzureOpenAI
from tqdm import tqdm

# ===================== 你原始的提示词 100% 不变 =====================
raw_prompt = """
You are tasked with conducting a careful examination of the provided image. Based on the content of the image, please answer the following yes or no questions:

Questions:
##YNQuestions##

Note that:
1. Each answer should be on a separate line, starting with "yes" or "no", followed by the reason.
2. The order of answers must correspond exactly to the order of the questions.
3. Each question must have only one answer.
4. Directly return the answers to each question, without any additional content.
5. Each answer must be on its own line!
6. Make sure the number of output answers equal to the number of questions!
"""

raw_prompt_1 = """
You are tasked with conducting a careful examination of the image. Based on the content of the image, please answer the following yes or no questions:

Questions:
##YNQuestions##

Note that:
Each answer should be on a separate line, starting with "yes" or "no", followed by the reason.
The order of answers must correspond exactly to the order of the questions.
Each question must have only one answer. Output one answer if there is only one question.
Directly return the answers to each question, without any additional content.
Each answer must be on its own line!
Make sure the number of output answers equal to the number of questions!
"""

raw_prompt_2 = """
You are tasked with carefully examining the provided image and answering the following yes or no questions:

Questions:
##YNQuestions##

Instructions:

1. Answer each question on a separate line, starting with "yes" or "no", followed by a brief reason.
2. Maintain the exact order of the questions in your answers.
3. Provide only one answer per question.
4. Return only the answers—no additional commentary.
5. Each answer must be on its own line.
6. Ensure the number of answers matches the number of questions.
"""


# ===================== 你原始的所有函数 100% 完全不变！！！ =====================
def load_jsonl_lines(jsonl_file):
    """读取 jsonl 文件，每行 parse 成 json 对象，返回列表"""
    lines = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                lines.append(obj)
            except Exception as e:
                print(f"[Warning] Parse line error in {jsonl_file}: {e}")
    return lines


def generate_with_prompt(prompt, image_path, client, model="gpt-4o"):
    # 完全和你原始代码一样！不做任何格式转换！
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")

    messages = [
        {"role": "system", "content": "You are a professional image critic."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
            ],
        },
    ]

    completion = client.chat.completions.create(model=model, messages=messages, temperature=1.0)

    return completion.choices[0].message.content


def format_questions_prompt(raw_prompt, questions):
    question_texts = [item.strip() for item in questions]
    formatted_questions = "\n".join(question_texts)
    prompt_template = random.choice([raw_prompt, raw_prompt_1, raw_prompt_2])
    formatted_prompt = prompt_template.replace("##YNQuestions##", formatted_questions)
    return formatted_prompt


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def find_image_by_idx(img_dir, idx):
    pattern = os.path.join(img_dir, f"{idx}.*")
    files = [f for f in glob.glob(pattern) if f.lower().endswith((".png", ".jpg", ".jpeg", "webp"))]
    if files:
        return files[0]
    else:
        raise FileNotFoundError(f"No image found for index {idx} in {img_dir}")


def collect_tasks(jsonl_dir, image_dir, eval_model, output_dir, sample_idx_file=None, postfix=""):
    tasks = []
    jsonl_files = glob.glob(os.path.join(jsonl_dir, "*.jsonl"))
    if sample_idx_file is not None:
        with open(sample_idx_file, "r") as f:
            sample_idx = json.load(f)
    for jsonl_file in jsonl_files:
        attribute = os.path.splitext(os.path.basename(jsonl_file))[0]
        lines = load_jsonl_lines(jsonl_file)
        attr_type = lines[0]["type"]
        if sample_idx_file is not None:
            line_indices = sample_idx[attr_type]
            lines = [lines[idx] for idx in line_indices]
        else:
            line_indices = list(range(len(lines)))
        for desc in ["long_description", "short_description"]:
            img_dir = os.path.join(image_dir, attr_type + postfix, eval_model, desc)
            out_dir = os.path.join(output_dir, eval_model, attr_type, "long" if desc.startswith("long") else "short")
            ensure_dir(out_dir)
            for idx, line in zip(line_indices, lines):
                try:
                    img_path = find_image_by_idx(img_dir, idx)
                    out_path = os.path.join(out_dir, f"{idx}.json")
                    if os.path.exists(out_path):
                        continue
                except Exception as e:
                    print(e)
                    continue
                tasks.append(
                    {
                        "attribute": attr_type,
                        "desc": desc,
                        "jsonl_file": jsonl_file,
                        "line_idx": idx,
                        "jsonl_line": line,
                        "img_path": img_path,
                        "out_path": out_path,
                    }
                )
    return tasks


class OutputFormatError(Exception):
    pass


def extract_yes_no(model_output, questions):
    lines = [line.strip() for line in model_output.strip().split("\n") if line.strip()]
    preds = []
    for idx, line in enumerate(lines):
        # m = re.match(r'^(yes|no)\b', line.strip(), flags=re.IGNORECASE)
        m = re.search(r"(yes|no)", line, re.IGNORECASE)
        if m:
            preds.append(m.group(1).lower())
        else:
            continue
    if len(preds) != len(questions):
        raise OutputFormatError(f"Preds count {len(preds)} != questions count {len(questions)}")
    return preds


# ===================== 线程处理函数（完全复用你原始逻辑） =====================
def process_task(task, args):
    try:
        # 每个线程独立创建 client，避免冲突
        client = AzureOpenAI(
            api_key=args.api_key,
            azure_endpoint=args.azure_endpoint,
            api_version=args.api_version,
        )

        item = task["jsonl_line"]
        questions = item.get("yn_question_list", [])
        gt_answers = item.get("yn_answer_list", [])
        prompt = format_questions_prompt(raw_prompt, questions)

        # 调用你原始的函数，完全不变
        model_output = generate_with_prompt(prompt, task["img_path"], client, model=args.model)
        model_pred = extract_yes_no(model_output, questions)

        result = {
            "attribute": task["attribute"],
            "desc": task["desc"],
            "jsonl_file": os.path.basename(task["jsonl_file"]),
            "line_idx": task["line_idx"],
            "questions": questions,
            "gt_answers": gt_answers,
            "model_pred": model_pred,
            "model_output": model_output,
        }

        with open(task["out_path"], "w", encoding="utf-8") as fout:
            json.dump(result, fout, ensure_ascii=False, indent=2)

        return True, task["out_path"], None

    except Exception as e:
        return False, task["img_path"], str(e)


# ===================== 8线程并行主函数 =====================
def main(args):
    tasks = collect_tasks(
        args.jsonl_dir, args.image_dir, args.eval_model, args.output_dir, args.sample_idx_file, args.postfix
    )
    print(f"Total tasks to process: {len(tasks)}")

    MAX_WORKERS = 64
    futures = []
    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        print(f"Start with {MAX_WORKERS} threads...")

        for task in tasks:
            futures.append(executor.submit(process_task, task, args))

        for future in tqdm(futures):
            ok, path, err = future.result()
            if ok:
                success += 1
            else:
                failed += 1
                print(f"[Failed] {path} | {err}")

    print(f"\nAll done! Success: {success} | Failed: {failed}")


# ===================== 入口 =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_dir", type=str, required=True, help="Directory containing jsonl files")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--eval_model", type=str, required=True, help="name of the eval model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output json files")
    parser.add_argument("--api_key", type=str, default="sk-xxx", help="Azure OpenAI API key")
    parser.add_argument(
        "--azure_endpoint",
        type=str,
        default="https://duomotai.openai.azure.com/openai/deployments/gpt-4.1/chat/completions?api-version=2025-01-01-preview",
        help="Azure OpenAI endpoint",
    )
    parser.add_argument("--api_version", type=str, default="2025-01-01-preview", help="Azure OpenAI API version")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Azure deployment/model name")
    parser.add_argument("--sample_idx_file", type=str, default=None, help="File containing sample indices")
    parser.add_argument("--postfix", type=str, default="")

    args = parser.parse_args()
    main(args)
