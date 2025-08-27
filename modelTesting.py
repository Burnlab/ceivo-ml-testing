import argparse
import json
from pathlib import Path
import random
import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3


def detect_image_format(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    return "jpeg" if ext == "jpg" else ext

def load_image_bytes(path: Path) -> bytes:
    return path.read_bytes()

def call_converse_once(client, model_id: str, prompt: str, image_bytes: bytes, image_format: str,
                       max_tokens: int = 400, temperature: float = 0.2, top_p: float = 0.9) -> dict:
    """
    Sends a single multimodal message (text + image) to a Bedrock model via Converse.
    Returns {"model_id": ..., "text": ..., "raw": <full response dict>}.
    """

    messages = [{
        "role": "user",
        "content": [
            {"text": prompt},
            {"image": {"format": image_format, "source": {"bytes": image_bytes}}},
        ],
    }]

    resp = client.converse(
        modelId=model_id,
        messages=messages,
    )

    # Extract concatenated text blocks from the first output message.
    content_blocks = resp.get("output", {}).get("message", {}).get("content", []) or []
    description = "".join(block.get("text", "") for block in content_blocks if "text" in block).strip()

    return {"model_id": model_id, "text": description, "raw": resp}

def call_openai(client, model_id: str, prompt: str, image_path: Path, image_format: str,
                max_tokens: int = 400, temperature: float = 0.2, top_p: float = 0.9) -> dict:
    """
    Sends a single multimodal message (text + image) to an OpenAI model.
    Returns {"model_id": ..., "text": ..., "raw": <full response dict>}.
    """
    image = client.files.create(
        file=open(image_path, "rb"),
        purpose="user_data"
    )
    messages = [{
        "role": "user",
        "content": 
        [
            {
                "type": "input_image",
                "file_id": image.id
            },
            {
                "type": "input_text",
                "text": prompt
            }
        ],
    }]

    resp = client.responses.create(
        model=model_id,
        input=messages,
    )


    return {"model_id": model_id, "text": resp.output_text, "raw": resp}

def random_image_path(folder):
    exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}
    images = [p for p in Path(folder).iterdir() if p.suffix.lower() in exts]
    return random.choice(images) if images else None

def test_prompt(prompt, image_path):
    model_ids = [
        "us.anthropic.claude-opus-4-1-20250805-v1:0"
    ]
    image_format = detect_image_format(image_path)
    image_bytes = load_image_bytes(image_path)
    region = "us-east-1"
    bedrock_client = boto3.client("bedrock-runtime", region_name=region)
    openai_client = OpenAI()

    max_tokens = 8191

    results = []
    with ThreadPoolExecutor(max_workers=min(8, len(model_ids))) as pool:
            futs = {
                pool.submit(
                    call_openai if m.startswith("gpt") else call_converse_once,
                    openai_client if m.startswith("gpt") else bedrock_client,
                    m, prompt, 
                    image_path if m.startswith("gpt") else image_bytes,
                    image_format, max_tokens, 0.2, 5
                ): m for m in model_ids
            }
            for fut in as_completed(futs):
                try:
                    result = fut.result()
                    if result["raw"] and "usage" in result["raw"]:
                        result["totalTokens"] = result["raw"]["usage"].get("totalTokens")

                    elif result["raw"].usage:
                        result["totalTokens"] = result["raw"].usage.total_tokens
                    else:
                        result["totalTokens"] = "N/A"
                    results.append(result)
                except Exception as e:
                    results.append({"model_id": futs[fut], "text": f"[ERROR] {type(e).__name__}: {e}", "raw": None, "totalTokens": "N/A"})

    from IPython.display import HTML, display
    from pathlib import Path


    img_src = Path(image_path).as_posix()


    json_output = {
       "image": img_src,
       "results": results
    }

    return json_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a prompt on an image folder")
    parser.add_argument("--folder", type=str, help="Path to the image folder.")
    args = parser.parse_args()

    image_folder = args.folder
    if not image_folder:
        print("No image folder provided.")
        exit(1)


    prompt = """You are an AI model designed to perform in-depth analysis of a single frame extracted from a video stream.
    The frame represents a clear image of a detected scene within the video. Your task is to analyze and describe the action, entities, and any relevant elements present in the image.
    Instructions:
    1. Provide a single paragraph summary that encapsulates the overall scene, including key actions, entities, and environmental context.
    2. Be specific with the names of any characters you recognize. If you do not recognize any characters say nothing on the subject 
    3. use simple but descriptive language
    4. Directly describe the scene without any introductory phrases or explanations, like "in the image" or "the scene shows."""

    json_results = []
    for image_path in Path(image_folder).iterdir():
        if image_path.is_file():
            json_results.append(test_prompt(prompt, image_path))
        print(json_results)

    with open("output.json", "w", encoding="utf-8") as f:
       json.dump(json_results, f, ensure_ascii=False, indent=2)