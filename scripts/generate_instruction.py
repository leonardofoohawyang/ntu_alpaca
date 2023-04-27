"""
run:
python scripts/generate_instruction.py generate_instruction_following_data --num_instructions_to_generate 5000
"""
import argparse
import json
import os
import random
import re
import time

import numpy as np
import tqdm
import utils
from ckip_transformers.nlp import CkipWordSegmenter
from rouge_chinese import Rouge

ws_driver = CkipWordSegmenter(model="bert-base", device=3)


def compute_rouge_l_f1_score(tokens1, tokens2):
    """Compute ROUGE-L F1 score."""
    scores = Rouge().get_scores(' '.join(tokens1), ' '.join(tokens2))
    return scores[0]['rouge-l']['f']


def encode_prompt_zh(prompt_instructions):
    """Encode multiple prompt instructions into a single string."""
    prompt = open("./data/prompt_zh.txt").read() + "\n"

    for idx, task_dict in enumerate(prompt_instructions):
        instruction, input = task_dict["instruction"], task_dict["input"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        input = "<noinput>" if input.lower() == "" else input
        prompt += f"###\n{idx + 1}. instruction: {instruction}\n{idx + 1}. input:\n{input}\n"
    prompt += f"###\n{idx + 2}. instruction:"
    return prompt


def post_process_response(num_prompt_instructions, response):
    if response is None:
        return []

    raw_instructions = f"{num_prompt_instructions+1}. instruction:" + \
        response["message"]["content"]
    raw_instructions = re.split("###", raw_instructions)
    instructions = []

    for idx, inst in enumerate(raw_instructions):
        if idx == len(raw_instructions) - 1 and response["finish_reason"] == "length":
            print(f"Discarded due to truncation:\n{inst}")
            continue

        idx += num_prompt_instructions + 1
        splitted_data = re.split(f"{idx}\.\s*(instruction|input):", inst)

        if len(splitted_data) != 5:
            print(f"Incomplete data:\n{splitted_data}")
            continue

        inst = splitted_data[2].strip()
        input = splitted_data[4].strip()
        input = "" if input.lower() == "<noinput>" else input

        if any(word in inst for word in get_blacklist()):
            print(f"Filtered:\n{inst}")
            continue

        instructions.append({"instruction": inst, "input": input})
    return instructions


def get_blacklist():
    return ["圖片", "圖表", "圖形", "檔案", "地圖", "繪圖", "繪制", "前往", "視頻", "影片", "音頻", "音樂", "流程圖", "示意圖"]


def load_seed_tasks(seed_tasks_path):
    return [json.loads(l) for l in open(seed_tasks_path, "r", encoding="utf-8")]


def load_instructions(seed_tasks):
    seed_instruction_data = []
    for t in seed_tasks:
        instance = t["instances"][0]
        if "output" in instance:
            seed_instruction_data.append(
                {
                    "instruction": t["instruction"],
                    "input": instance["input"],
                }
            )
    return seed_instruction_data


def load_machine_instructions(output_dir):
    machine_instruction_data = []
    if os.path.exists(os.path.join(output_dir, "regen.json")):
        machine_instruction_data = utils.jload(
            os.path.join(output_dir, "regen.json"))
    return machine_instruction_data


def generate_instructions(args):
    seed_tasks = load_seed_tasks(args.seed_tasks_path)
    seed_instruction_data = load_instructions(seed_tasks)
    machine_instruction_data = load_machine_instructions(args.output_dir)

    progress_bar = tqdm.tqdm(total=args.num_instructions_to_generate)
    progress_bar.update(len(machine_instruction_data))

    all_instructions = [d["instruction"] for d in seed_instruction_data] + \
        [d["instruction"] for d in machine_instruction_data]

    all_instruction_tokens = ws_driver(all_instructions)

    while len(machine_instruction_data) < args.num_instructions_to_generate:
        decoding_args = utils.OpenAIDecodingArguments(
            temperature=args.temperature,
            n=1,
            max_tokens=4096,
            top_p=args.top_p,
            stop=["\n20", "20.", "20."],
        )

        prompt_instructions = random.sample(
            seed_instruction_data, args.num_prompt_instructions)
        prompt = encode_prompt_zh(prompt_instructions)
        messages = [{"role": "system", "content": f"Task: {prompt}"}]

        results = utils.completions_with_backoff(
            messages=messages,
            model_name=args.model_name,
            decoding_args=decoding_args,
            logit_bias={"50256": -100},
        )

        instruction_data = []
        for result in results:
            new_instructions = post_process_response(
                args.num_prompt_instructions, result)
            instruction_data += new_instructions

        total = len(instruction_data)
        keep = 0

        # Display the content of the generated instructions
        print(instruction_data)

        for instruction_data_entry in instruction_data:
            new_instruction_tokens = ws_driver(
                [instruction_data_entry["instruction"]])[0]

            print(new_instruction_tokens)

            rouge_scores = [
                compute_rouge_l_f1_score(new_instruction_tokens, tokens)
                for tokens in all_instruction_tokens
            ]

            most_similar_instructions = {
                all_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
            }
            if max(rouge_scores) > 0.7:
                continue
            else:
                keep += 1

            instruction_data_entry["most_similar_instructions"] = most_similar_instructions
            instruction_data_entry["avg_similarity_score"] = float(
                np.mean(rouge_scores))
            machine_instruction_data.append(instruction_data_entry)
            all_instructions.append(instruction_data_entry["instruction"])

            all_instruction_tokens.append(new_instruction_tokens)
            progress_bar.update(1)

        print(f"Generated {total} instructions, kept {keep} instructions")
        utils.jdump(machine_instruction_data, os.path.join(
            args.output_dir, "regen.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate instructions using GPT-3")
    parser.add_argument(
        "task", type=str, help="The task to perform: generate_instruction_following_data")
    parser.add_argument("--output_dir", type=str, default="data/",
                        help="Output directory for the generated instructions")
    parser.add_argument("--seed_tasks_path", type=str,
                        default="data/seed_tasks_zh.jsonl", help="Path to the seed tasks JSONL file")
    parser.add_argument("--num_instructions_to_generate", type=int,
                        default=5000, help="Number of instructions to generate")
    parser.add_argument("--model_name", type=str,
                        default="gpt-4", help="Name of the GPT-3 model to use")
    parser.add_argument("--num_prompt_instructions", type=int,
                        default=3, help="Number of prompt instructions to use")
    parser.add_argument("--request_batch_size", type=int,
                        default=5, help="Batch size for GPT-3 requests")
    parser.add_argument("--temperature", type=float,
                        default=0.7, help="Temperature for GPT-3 sampling")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top p for GPT-3 sampling")

    args = parser.parse_args()

    if args.task == "generate_instruction_following_data":
        generate_instructions(args)
    else:
        print(f"Unknown task: {args.task}")
