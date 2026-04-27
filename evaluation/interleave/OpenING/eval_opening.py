import argparse
import base64
import csv
import glob
import io
import json
import os
import random
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import openai  # Make sure to install this package
import pandas as pd
import requests
from openai import OpenAI
from PIL import Image, ImageDraw
from tqdm import tqdm

# Runtime configuration
API_BASE_URL = os.getenv("OPENING_JUDGE_BASE_URL", "")
API_KEY = os.getenv("OPENING_JUDGE_API_KEY", os.getenv("OPENAI_API_KEY", ""))
JUDGE_MODEL = os.getenv("OPENING_JUDGE_MODEL", "gpt-4o")
CHAT_COMPLETIONS_URL = ""

ROOT_DIR = os.path.abspath(os.getenv("OPENING_EVAL_ROOT", os.getcwd()))
OPENING_DIR = ""
PK_FILE_NAME = ""
OUTPUT_FILE = ""
PROMPT_FILE = ""
EXPECTED_SCORE_METRICS = [
    "Correctness",
    "Image-Text Coherency",
    "Multi-step Consistency",
    "Content Quality",
    "Human Preference Alignment",
    "Completeness",
    "Content Richness",
]
NUM_RETRIES = int(os.getenv("GPT_SCORE_NUM_RETRIES", "3"))
REQUEST_TIMEOUT = float(os.getenv("GPT_SCORE_REQUEST_TIMEOUT", "180"))
DEBUG = os.getenv("GPT_SCORE_DEBUG", "0") == "1"
DEFAULT_WORKERS = int(os.getenv("GPT_SCORE_WORKERS", "4"))
DEFAULT_SAVE_EVERY = int(os.getenv("GPT_SCORE_SAVE_EVERY", "10"))
SESSION_LOCAL = threading.local()
SAVE_LOCK = threading.Lock()


def resolve_opening_dir(opening_root, benchmark_dir=None):
    candidates = []
    if benchmark_dir:
        candidates.append(os.path.abspath(benchmark_dir))
    candidates.extend(
        [
            os.path.join(opening_root, "OpenING-benchmark"),
            os.path.join(opening_root, "OpenING-Benchmark"),
            opening_root,
        ]
    )
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "test_data.jsonl")):
            return candidate
    return candidates[0]


def configure_runtime(args):
    global API_BASE_URL, API_KEY, JUDGE_MODEL, CHAT_COMPLETIONS_URL
    global ROOT_DIR, OPENING_DIR, PK_FILE_NAME, OUTPUT_FILE, PROMPT_FILE

    ROOT_DIR = os.path.abspath(args.opening_root)
    OPENING_DIR = resolve_opening_dir(ROOT_DIR, args.benchmark_dir)
    PK_FILE_NAME = os.path.abspath(
        args.pk_file or os.path.join(ROOT_DIR, "Interleaved_Arena", "data_instance_modelAB_new.json")
    )
    OUTPUT_FILE = os.path.abspath(
        args.output_file or os.path.join(ROOT_DIR, "Interleaved_Arena", "gpt-score_results_new.json")
    )
    prompt_candidates = []
    if args.prompt_file:
        prompt_candidates.append(os.path.abspath(args.prompt_file))
    prompt_candidates.extend(
        [
            os.path.join(ROOT_DIR, "prompts", "detailed_score_system.txt"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", "detailed_score_system.txt"),
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts", "detailed_score_system.txt"
            ),
        ]
    )
    PROMPT_FILE = prompt_candidates[0]
    for candidate in prompt_candidates:
        if os.path.exists(candidate):
            PROMPT_FILE = candidate
            break

    API_BASE_URL = args.api_base_url or API_BASE_URL
    API_KEY = args.api_key or API_KEY
    JUDGE_MODEL = args.judge_model or JUDGE_MODEL
    CHAT_COMPLETIONS_URL = f"{API_BASE_URL.rstrip('/')}/v1/chat/completions" if API_BASE_URL else ""


def validate_runtime(args):
    if args.mode == "pairwise_file" and not os.path.exists(PK_FILE_NAME):
        raise FileNotFoundError(f"pairwise PK file not found: {PK_FILE_NAME}")

    test_data_path = os.path.join(OPENING_DIR, "test_data.jsonl")
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"OpenING benchmark test_data.jsonl not found under: {OPENING_DIR}")
    if not os.path.exists(PROMPT_FILE):
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_FILE}")

    if not API_BASE_URL:
        raise ValueError("Missing judge API base URL. Use --api_base_url or OPENING_JUDGE_BASE_URL.")
    if not API_KEY:
        raise ValueError("Missing judge API key. Use --api_key or OPENING_JUDGE_API_KEY / OPENAI_API_KEY.")


def load_judge_results():
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            judge_results = json.load(f)
    else:
        judge_results = []
    return judge_results


def load_pk_file():
    with open(os.path.join(PK_FILE_NAME), "r") as f:
        pk_data = json.load(f)
        pk_list = []
        for data in pk_data:
            pk_list.append(data)
        return pk_list


def parse_and_load_json(content):
    input_text_list = []
    input_image_list = []
    onput_text_list = []
    output_image_list = []

    for input_step, input_content in enumerate(content["conversations"][0]["input"]):
        input_text_list.append(input_content["text"].strip())
        input_image_list.append(input_content["image"])

    for output_step, output_content in enumerate(content["conversations"][1]["output"]):
        onput_text_list.append(output_content["text"].strip())
        output_image_list.append(output_content["image"])

    return input_text_list, input_image_list, onput_text_list, output_image_list


def load_data(data_path):
    with open(data_path, encoding="utf-8") as file:  # 打开数据文件
        content = json.load(file)
        ori_data = content  # 将每行数据加载为JSON对象并添加到列表
        # get input list etc. and return 5 lists
        ainput_list, ainput_image_list, aoutput_list, aoutput_image_list = parse_and_load_json(content)
        io_data = {
            "input_text": ainput_list,
            "input_image": ainput_image_list,
            "output_text": aoutput_list,
            "output_image": aoutput_image_list,
        }
    return ori_data, io_data


def get_request_session():
    session = getattr(SESSION_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        SESSION_LOCAL.session = session
    return session


def build_judge_messages(
    input_text,
    input_images,
    output_text_list,
    output_images,
    include_input_images=True,
    include_output_images=True,
    note_text=None,
):
    my_message = [{"role": "system", "content": SYSTEM_MESSAGE}]

    content = []
    input_text[0] = "INPUT: " + input_text[0]
    for i in range(len(input_text)):
        content.append({"type": "text", "text": input_text[i].replace("<BEGIN>", "")})
        if include_input_images and i < len(input_images) and input_images[i] != None:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{input_images[i]}"}})

    if len(output_text_list) >= 1 and output_text_list[0]:
        output_text_list[0] = "\nOUTPUT: " + output_text_list[0]
    elif len(output_text_list) >= 1:
        output_text_list[0] = "\nOUTPUT: "
    else:
        output_text_list.append("\nOUTPUT: None")

    for i in range(max(len(output_text_list), len(output_images))):
        if i < len(output_text_list):
            content.append({"type": "text", "text": output_text_list[i]})
        if include_output_images and i < len(output_images) and output_images[i] != None:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{output_images[i]}"}})

    if note_text:
        content.append({"type": "text", "text": f"\nNote: {note_text}"})

    content.append({"type": "text", "text": "\nPlease only output the json result: "})

    my_message.append({"role": "user", "content": content})
    return my_message


def get_gpt4answer(input_text, input_images, modelA_output_textl, modelA_output_images):
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
        "Content-Type": "application/json",
    }

    input_text = list(input_text)
    modelA_output_textl = list(modelA_output_textl)

    my_message = build_judge_messages(
        list(input_text),
        list(input_images),
        list(modelA_output_textl),
        list(modelA_output_images),
        include_input_images=True,
        include_output_images=True,
    )

    functions = [
        {
            "name": "evaluate_multimodal_content",
            "description": "Evaluate the quality of interleaved image-text content based on specific criteria",
            "parameters": {
                "type": "object",
                "properties": {
                    "Correctness": {
                        "type": "object",
                        "properties": {"Score": {"type": "integer"}, "Justification": {"type": "string"}},
                        "required": ["Score", "Justification"],
                    },
                    "Image-Text Coherency": {
                        "type": "object",
                        "properties": {"Score": {"type": "integer"}, "Justification": {"type": "string"}},
                        "required": ["Score", "Justification"],
                    },
                    "Multi-step Consistency": {
                        "type": "object",
                        "properties": {"Score": {"type": "integer"}, "Justification": {"type": "string"}},
                        "required": ["Score", "Justification"],
                    },
                    "Content Quality": {
                        "type": "object",
                        "properties": {"Score": {"type": "integer"}, "Justification": {"type": "string"}},
                        "required": ["Score", "Justification"],
                    },
                    "Human Preference Alignment": {
                        "type": "object",
                        "properties": {"Score": {"type": "integer"}, "Justification": {"type": "string"}},
                        "required": ["Score", "Justification"],
                    },
                    "Completeness": {
                        "type": "object",
                        "properties": {"Score": {"type": "integer"}, "Justification": {"type": "string"}},
                        "required": ["Score", "Justification"],
                    },
                    "Content Richness": {
                        "type": "object",
                        "properties": {"Score": {"type": "integer"}, "Justification": {"type": "string"}},
                        "required": ["Score", "Justification"],
                    },
                },
                "required": [
                    "Correctness",
                    "Image-Text Coherency",
                    "Multi-step Consistency",
                    "Content Quality",
                    "Human Preference Alignment",
                    "Completeness",
                    "Content Richness",
                ],
            },
        }
    ]

    payload = {
        "model": JUDGE_MODEL,
        "messages": my_message,
        "functions": functions,
        "function_call": {"name": "evaluate_multimodal_content"},
    }

    final_answer = ""
    last_error = None
    for attempt in range(NUM_RETRIES):
        try:
            response = get_request_session().post(
                CHAT_COMPLETIONS_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            if response.status_code == 200:
                response_json = response.json()
                message = response_json["choices"][0]["message"]
                if DEBUG:
                    print(message)

                if "function_call" in message:
                    answer = message["function_call"]["arguments"]
                else:
                    answer = message.get("content", "")

                final_answer = normalize_judge_score(json.loads(answer))
                if final_answer and is_valid_score(final_answer):
                    return final_answer
                if final_answer:
                    last_error = f"Invalid judge result schema: {list(final_answer.keys()) if isinstance(final_answer, dict) else type(final_answer).__name__}"
                    print(f"API request failed: {last_error}")
                    continue

                last_error = "Empty judge result"
            else:
                try:
                    error_payload = response.json()
                except Exception:
                    error_payload = {}

                error_code = (error_payload.get("error") or {}).get("code") if isinstance(error_payload, dict) else None
                if error_code == "content_policy_violation":
                    print("Content policy violation during GPT judge; retrying text-only fallback")
                    fallback_messages = build_judge_messages(
                        list(input_text),
                        list(input_images),
                        list(modelA_output_textl),
                        list(modelA_output_images),
                        include_input_images=False,
                        include_output_images=False,
                        note_text="Images were omitted because the automated judge rejected them under its content safety policy. Evaluate based on the textual instruction and textual output only, and reflect any uncertainty in the justifications.",
                    )
                    fallback_payload = {
                        "model": JUDGE_MODEL,
                        "messages": fallback_messages,
                        "functions": functions,
                        "function_call": {"name": "evaluate_multimodal_content"},
                    }
                    fallback_response = get_request_session().post(
                        CHAT_COMPLETIONS_URL,
                        headers=headers,
                        json=fallback_payload,
                        timeout=REQUEST_TIMEOUT,
                    )
                    if fallback_response.status_code == 200:
                        fallback_json = fallback_response.json()
                        fallback_message = fallback_json["choices"][0]["message"]
                        if DEBUG:
                            print(fallback_message)
                        if "function_call" in fallback_message:
                            fallback_answer = fallback_message["function_call"]["arguments"]
                        else:
                            fallback_answer = fallback_message.get("content", "")
                        final_answer = normalize_judge_score(json.loads(fallback_answer))
                        if final_answer and is_valid_score(final_answer):
                            return final_answer
                        if final_answer:
                            last_error = f"Invalid text-only fallback schema: {list(final_answer.keys()) if isinstance(final_answer, dict) else type(final_answer).__name__}"
                            print(f"API request failed: {last_error}")
                            break
                    last_error = f"text_only_fallback_status={fallback_response.status_code}, body={fallback_response.text[:500]}"
                    print(f"API request failed: {last_error}")
                    break

                last_error = f"status_code={response.status_code}, body={response.text[:500]}"
                print(f"API request failed: {last_error}")
        except Exception as e:
            last_error = e
            print(f"API request failed: {e}")

        if attempt + 1 < NUM_RETRIES:
            time.sleep(1)

    print(f"Failed to get GPT judge result after {NUM_RETRIES} retries: {last_error}")
    return False


# Function to encode the image
@lru_cache(maxsize=8192)
def encode_image(image_path):
    # Open the image file
    with Image.open(image_path) as img:
        # Resize the image to 512x512, maintaining aspect ratio if necessary
        img = img.resize((512, 512))
        # JPEG does not support alpha channels like LA/RGBA, so flatten them first.
        if img.mode in {"RGBA", "LA"}:
            alpha = img.getchannel("A")
            rgb_img = Image.new("RGB", img.size, (255, 255, 255))
            rgb_img.paste(img.convert("RGBA").convert("RGB"), mask=alpha)
            img = rgb_img
        elif img.mode == "P":
            img = img.convert("RGBA")
            alpha = img.getchannel("A")
            rgb_img = Image.new("RGB", img.size, (255, 255, 255))
            rgb_img.paste(img.convert("RGB"), mask=alpha)
            img = rgb_img
        elif img.mode != "RGB":
            img = img.convert("RGB")
        # Save the image to a bytes buffer in JPEG format
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        img_bytes = buffer.getvalue()

        # Encode the image to base64
        return base64.b64encode(img_bytes).decode("utf-8")


def decode_image(encoded_string):
    # 解码 base64 字符串
    image_data = base64.b64decode(encoded_string)
    # 将解码后的数据转换为图像格式
    image = Image.open(BytesIO(image_data))
    image.save("temp.jpg")
    return image


def normalize_judge_score(score):
    if not isinstance(score, dict):
        return score

    wrapped_scores = score.get("scores")
    if isinstance(wrapped_scores, dict):
        return wrapped_scores

    return score


def save_judge_results(judge_results):
    output_dir = os.path.dirname(os.path.abspath(OUTPUT_FILE))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with SAVE_LOCK:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(judge_results, f, ensure_ascii=False, indent=4)


def is_valid_score(score):
    score = normalize_judge_score(score)
    if not isinstance(score, dict):
        return False
    for metric in EXPECTED_SCORE_METRICS:
        metric_data = score.get(metric)
        if not isinstance(metric_data, dict):
            return False
        metric_score = metric_data.get("Score")
        if metric_score is None or metric_score == "":
            return False
        if not isinstance(metric_score, (int, float)):
            return False
    return True


def is_valid_judge_result(result):
    return isinstance(result, dict) and is_valid_score(result.get("score"))


def get_output_dir_result_key(result):
    model = result.get("model", {}) if isinstance(result, dict) else {}
    model_name = model.get("name") or model.get("id")
    instance_id = (result.get("instance_id") or result.get("data_id")) if isinstance(result, dict) else None
    if not model_name or not instance_id:
        return None
    return str(instance_id), str(model_name)


def upsert_judge_result(judge_results, result_data):
    result_key = get_output_dir_result_key(result_data)
    if result_key is None:
        judge_results.append(result_data)
        return

    new_results = []
    replaced = False
    for existing_result in judge_results:
        if get_output_dir_result_key(existing_result) == result_key:
            if not replaced:
                new_results.append(result_data)
                replaced = True
            continue
        new_results.append(existing_result)

    if not replaced:
        new_results.append(result_data)

    judge_results[:] = new_results


def discover_model_output_dirs(output_dir):
    """
    Accept either:
    1. a parent directory containing multiple *_output folders, or
    2. one specific *_output folder containing JSON result files.
    """
    if os.path.isdir(output_dir) and os.path.basename(output_dir).endswith("_output"):
        return [(os.path.basename(output_dir), output_dir)]

    model_dirs = []
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item.endswith("_output"):
            model_dirs.append((item, item_path))
    return model_dirs


def evaluate_output_instance(model_name, model_output_dir, instance_id, model_output):
    input_text_list, input_image_list, output_text_list, output_image_list = parse_and_load_json(model_output)

    input_images = []
    img_count = -1
    input_img_count = -1

    for i, img_path in enumerate(input_image_list):
        if img_path:
            full_img_path = os.path.join(OPENING_DIR, img_path)
            if os.path.exists(full_img_path):
                try:
                    input_images.append(encode_image(full_img_path))
                    img_count += 1
                    input_img_count += 1
                    input_text_list[i] = (
                        input_text_list[i].replace("<BEGIN>", "").replace("<image>", "")
                        + f" <IMG_{img_count}>"
                        + f"</IMG_{img_count}>"
                    )
                except Exception as e:
                    print(f"ERROR: {e}")
                    print(full_img_path)
                    input_images.append(None)
                    input_text_list[i] = input_text_list[i].replace("<BEGIN>", "").replace("<image>", "")
            else:
                input_images.append(None)
                input_text_list[i] = input_text_list[i].replace("<BEGIN>", "").replace("<image>", "")
        else:
            input_images.append(None)
            input_text_list[i] = input_text_list[i].replace("<BEGIN>", "").replace("<image>", "")

    output_images = []

    for i, text in enumerate(output_text_list):
        if output_image_list[i] is not None:
            img_path = output_image_list[i]
            full_img_path = os.path.join(model_output_dir, img_path.split("/")[-1])
            if os.path.exists(full_img_path):
                try:
                    output_images.append(encode_image(full_img_path))
                    input_img_count += 1
                    output_text_list[i] = (
                        text.replace("<image>", "") + f" <IMG_{input_img_count}>" + f"</IMG_{input_img_count}>"
                    )
                except Exception as e:
                    print(f"ERROR: {e}")
                    print(full_img_path)
                    output_text_list[i] = text.replace("<image>", "")
                    output_images.append(None)
            else:
                output_text_list[i] = text.replace("<image>", "")
                output_images.append(None)
        else:
            output_text_list[i] = text.replace("<image>", "")
            output_images.append(None)

    score = get_gpt4answer(input_text_list, input_images, output_text_list, output_images)
    return {"instance_id": instance_id, "model": {"name": model_name.replace("_output", "")}, "score": score}


def main_gpt_judge_score_by_pairwise_file_format():
    total_pk_list = load_pk_file()
    judge_results = load_judge_results()
    runned_id = []
    for i in judge_results:
        runned_id.append((i["data_id"], i["model"]["id"]))

    for index, pk_data in enumerate(total_pk_list):
        current_data_uid = pk_data["data_id"]

        if (current_data_uid, pk_data["model_A"]["id"]) in runned_id and (
            current_data_uid,
            pk_data["model_B"]["id"],
        ) in runned_id:
            continue

        current_model_A = pk_data["model_A"]["name"]
        current_model_B = pk_data["model_B"]["name"]
        current_file_path1 = os.path.join(ROOT_DIR, f"{current_model_A}_output", f"{current_data_uid}.json")
        current_file_path2 = os.path.join(ROOT_DIR, f"{current_model_B}_output", f"{current_data_uid}.json")

        try:
            ori_data1, a_data1 = load_data(current_file_path1)
            ori_data2, a_data2 = load_data(current_file_path2)
        except Exception as e:
            print(f"ERROR: {e}")
            print(current_file_path1, current_file_path2)
            continue

        # if len(a_data1['output_text']) < 1 or len(a_data2['output_text']) < 1:
        #     continue

        if a_data1["input_image"] != a_data2["input_image"]:
            if len(a_data1["input_image"]) > len(a_data2["input_image"]):
                input_image_list = a_data1["input_image"]
            else:
                input_image_list = a_data2["input_image"]
        else:
            input_image_list = a_data1["input_image"]
        assert a_data1["input_text"] == a_data2["input_text"]

        input_text_list = a_data1["input_text"]

        img_count = -1
        input_img_count = -1

        modelA_output_images = []
        modelB_output_images = []
        input_images = []

        for i in range(len(input_text_list)):
            # For each input text, get the GPT-4 generated answer
            img_path = a_data1["input_image"][i]
            if img_path:
                temp_img_path = os.path.join(OPENING_DIR, img_path)
                # if len(img_path) > 16:
                #     temp_img_path = os.path.join(OPENING_DIR, img_path)
                # else:
                #     temp_img_path = os.path.join(INPUT_DIR, img_path)
                try:
                    input_images.append(encode_image(temp_img_path))
                    img_count += 1
                    input_img_count += 1
                    input_text_list[i] = (
                        input_text_list[i].replace("<BEGIN>", "").replace("<image>", "")
                        + f" <IMG_{img_count}>"
                        + f"</IMG_{img_count}>"
                    )
                except Exception as e:
                    print(f"ERROR: {e}")
                    print(temp_img_path)
                    input_images.append(None)
                    input_text_list[i] = input_text_list[i].replace("<BEGIN>", "").replace("<image>", "")
            else:
                input_images.append(None)
                input_text_list[i] = input_text_list[i].replace("<BEGIN>", "").replace("<image>", "")

        for i in range(len(a_data1["output_text"])):
            if a_data1["output_image"][i] != None:
                temp_img_path = os.path.join(
                    ROOT_DIR, f"{current_model_A}_output", a_data1["output_image"][i].split("/")[-1]
                )
                if os.path.exists(temp_img_path):
                    try:
                        modelA_output_images.append(encode_image(temp_img_path))
                        img_count += 1
                        a_data1["output_text"][i] = (
                            a_data1["output_text"][i].replace("<image>", "")
                            + f" <IMG_{img_count}>"
                            + f"</IMG_{img_count}>"
                        )
                    except Exception as e:
                        print(f"ERROR: {e}")
                        print(temp_img_path)
                        a_data1["output_text"][i] = a_data1["output_text"][i].replace("<image>", "")
                        modelA_output_images.append(None)
                else:
                    a_data1["output_text"][i] = a_data1["output_text"][i].replace("<image>", "")
                    modelA_output_images.append(None)
            else:
                a_data1["output_text"][i] = a_data1["output_text"][i].replace("<image>", "")
                modelA_output_images.append(None)

        for i in range(len(a_data2["output_text"])):
            if a_data2["output_image"][i] != None:
                temp_img_path = os.path.join(
                    ROOT_DIR, f"{current_model_B}_output", a_data2["output_image"][i].split("/")[-1]
                )
                if os.path.exists(temp_img_path):
                    try:
                        modelB_output_images.append(encode_image(temp_img_path))
                        input_img_count += 1
                        a_data2["output_text"][i] = (
                            a_data2["output_text"][i].replace("<image>", "")
                            + f" <IMG_{input_img_count}>"
                            + f"</IMG_{input_img_count}>"
                        )
                    except Exception as e:
                        print(f"ERROR: {e}")
                        print(temp_img_path)
                        a_data2["output_text"][i] = a_data2["output_text"][i].replace("<image>", "")
                        modelB_output_images.append(None)
                else:
                    a_data2["output_text"][i] = a_data2["output_text"][i].replace("<image>", "")
                    modelB_output_images.append(None)
            else:
                a_data2["output_text"][i] = a_data2["output_text"][i].replace("<image>", "")
                modelB_output_images.append(None)

        output = get_gpt4answer(input_text_list, input_images, a_data1["output_text"], modelA_output_images)
        print(f"A: {output}")
        current_data = pk_data.copy()
        # delete model_B key and its value
        current_data.pop("model_B", None)
        # rename the key model_A as model
        current_data["model"] = current_data.pop("model_A")
        current_data["score"] = output
        judge_results.append(current_data)

        output = get_gpt4answer(input_text_list, input_images, a_data2["output_text"], modelB_output_images)
        print(f"B: {output}")
        current_data = pk_data.copy()
        # delete model_A key and its value
        current_data.pop("model_A", None)
        # rename the key model_B as model
        current_data["model"] = current_data.pop("model_B")
        current_data["score"] = output
        judge_results.append(current_data)

        save_judge_results(judge_results)


def main_evaluate_output_directory(
    output_dir,
    resume=True,
    limit=None,
    retry_invalid_scores=False,
    workers=DEFAULT_WORKERS,
    save_every=DEFAULT_SAVE_EVERY,
):
    """
    Evaluate all model outputs in the specified directory using GPT scoring.

    Args:
        output_dir (str): Directory containing model output folders, or one *_output folder
    """
    judge_results = load_judge_results() if resume else []
    completed_keys = set()
    for result in judge_results:
        key = get_output_dir_result_key(result)
        if key is None:
            continue
        if retry_invalid_scores and not is_valid_judge_result(result):
            continue
        completed_keys.add(key)

    model_dirs = discover_model_output_dirs(output_dir)
    print(f"Found {len(model_dirs)} model output directories: {[name for name, _ in model_dirs]}")
    print(f"Loaded existing judge results: {len(judge_results)}; resume={resume}")
    print(f"GPT judge workers: {workers}; save_every={save_every}")
    if retry_invalid_scores:
        invalid_existing = sum(
            1
            for result in judge_results
            if get_output_dir_result_key(result) is not None and not is_valid_judge_result(result)
        )
        print(f"Retry invalid scores: enabled; invalid existing results={invalid_existing}")

    # Load test instances: key is uid, value is instance
    test_instances = load_test_instances()

    if limit is not None and limit <= 0:
        print("Pending GPT judge tasks: 0")
        save_judge_results(judge_results)
        return

    tasks = []

    for model_name, model_output_dir in model_dirs:
        # Load model outputs
        model_outputs = load_model_outputs(model_output_dir)

        for instance_id in test_instances.keys():
            # Get model output for this instance
            if instance_id not in model_outputs:
                print(f"Warning: No output found for instance {instance_id} in {model_name}")
                continue

            result_key = (str(instance_id), model_name.replace("_output", ""))
            if resume and result_key in completed_keys:
                continue

            tasks.append((model_name, model_output_dir, instance_id, model_outputs[instance_id]))
            if limit is not None and len(tasks) >= limit:
                break

        if limit is not None and len(tasks) >= limit:
            break

    print(f"Pending GPT judge tasks: {len(tasks)}")
    if not tasks:
        save_judge_results(judge_results)
        return

    completed_since_save = 0
    failed_count = 0

    if workers <= 1:
        for model_name, model_output_dir, instance_id, model_output in tqdm(tasks, desc="GPT scoring"):
            try:
                result_data = evaluate_output_instance(model_name, model_output_dir, instance_id, model_output)
                print(f"{model_name} - Instance {instance_id}: {result_data['score']}")
                upsert_judge_result(judge_results, result_data)
                completed_keys.add(get_output_dir_result_key(result_data))
                completed_since_save += 1
                if completed_since_save >= save_every:
                    save_judge_results(judge_results)
                    completed_since_save = 0
            except Exception as e:
                print(f"Error evaluating {model_name} for instance {instance_id}: {e}")
                failed_count += 1
                continue
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_task = {
                executor.submit(evaluate_output_instance, model_name, model_output_dir, instance_id, model_output): (
                    model_name,
                    instance_id,
                )
                for model_name, model_output_dir, instance_id, model_output in tasks
            }

            for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="GPT scoring"):
                model_name, instance_id = future_to_task[future]
                try:
                    result_data = future.result()
                    print(f"{model_name} - Instance {instance_id}: {result_data['score']}")
                    upsert_judge_result(judge_results, result_data)
                    completed_keys.add(get_output_dir_result_key(result_data))
                    completed_since_save += 1
                    if completed_since_save >= save_every:
                        save_judge_results(judge_results)
                        completed_since_save = 0
                except Exception as e:
                    print(f"Error evaluating {model_name} for instance {instance_id}: {e}")
                    failed_count += 1

    save_judge_results(judge_results)
    print(
        f"GPT scoring finished. saved_results={len(judge_results)}, failed_tasks={failed_count}, output_file={OUTPUT_FILE}"
    )


def load_model_outputs(model_output_dir):
    """
    Load model outputs from the specified directory.

    Args:
        model_output_dir (str): Path to model output directory

    Returns:
        dict: Dictionary of model outputs by instance_id
    """
    model_outputs = {}

    # Look for output files in the directory
    output_files = glob.glob(os.path.join(model_output_dir, "*.json"))

    for output_file in output_files:
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                output_data = json.load(f)
                instance_uid = os.path.basename(output_file).split(".")[0]

                # Handle different output formats
                model_outputs[instance_uid] = output_data
        except Exception as e:
            print(f"Error loading output file {output_file}: {e}")
            continue
    return model_outputs


def load_test_instances():
    """
    Load test instances from the OpenING dataset.

    Returns:
        dict: Dictionary of test instances
    """
    test_instances = {}

    # Load test data from OpenING-Benchmark directory
    test_data_path = os.path.join(OPENING_DIR, "test_data.jsonl")
    if os.path.exists(test_data_path):
        with open(test_data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    instance = json.loads(line)
                    test_instances[instance["total_uid"]] = instance
    else:
        print(f"Warning: Test data file not found at {test_data_path}")

    return test_instances


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT Score Evaluation")
    parser.add_argument(
        "--mode",
        type=str,
        default="output_dir",
        choices=["pairwise_file", "output_dir"],
        help="Evaluation mode: pairwise_file or output_dir",
    )
    parser.add_argument("--opening_root", type=str, default=os.getcwd(), help="Root directory of the OpenING workspace")
    parser.add_argument(
        "--benchmark_dir", type=str, default=None, help="Optional explicit path to OpenING-benchmark directory"
    )
    parser.add_argument("--pk_file", type=str, default=None, help="Optional explicit path to pairwise PK JSON file")
    parser.add_argument(
        "--prompt_file", type=str, default=None, help="Optional explicit path to detailed_score_system.txt"
    )
    parser.add_argument(
        "--output_dir", type=str, default="gen_outputs", help="Directory containing model outputs for evaluation"
    )
    parser.add_argument("--output_file", type=str, default=None, help="Path to save GPT score results JSON")
    parser.add_argument("--api_base_url", type=str, default=None, help="Judge API base URL")
    parser.add_argument("--api_key", type=str, default=None, help="Judge API key")
    parser.add_argument("--judge_model", type=str, default=JUDGE_MODEL, help="Judge model name")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of concurrent GPT judge requests for output_dir mode",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=DEFAULT_SAVE_EVERY,
        help="Save results after this many completed GPT judge requests",
    )
    parser.add_argument("--no_resume", action="store_true", help="Do not skip samples already present in --output_file")
    parser.add_argument(
        "--retry_invalid_scores",
        action="store_true",
        help="Retry existing results whose score metrics are missing, null, or non-numeric",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only score the first N pending tasks, useful for smoke tests"
    )

    args = parser.parse_args()
    configure_runtime(args)
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.save_every < 1:
        raise ValueError("--save_every must be >= 1")
    validate_runtime(args)

    print(f"GPT score mode: {args.mode}")
    print(f"OpenING root: {ROOT_DIR}")
    print(f"Benchmark dir: {OPENING_DIR}")
    print(f"Prompt file: {PROMPT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")

    with open(PROMPT_FILE, "r") as f:
        file_contents = f.read()
    SYSTEM_MESSAGE = str(file_contents)

    if args.mode == "pairwise_file":
        # Use original logic for pairwise file evaluation
        main_gpt_judge_score_by_pairwise_file_format()
    elif args.mode == "output_dir":
        if args.output_dir is None:
            print("Error: --output_dir must be specified when using output_dir mode")
            exit()
        # New logic for output directory evaluation
        main_evaluate_output_directory(
            args.output_dir,
            resume=not args.no_resume,
            limit=args.limit,
            retry_invalid_scores=args.retry_invalid_scores,
            workers=args.workers,
            save_every=args.save_every,
        )
    else:
        print("Invalid mode specified. Use 'pairwise_file' or 'output_dir'")
