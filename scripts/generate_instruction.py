"""
batch_selfinstruct_generate.py

run:
python -m generate_instruction generate_instruction_following_data \
  --output_dir ./ \
  --num_instructions_to_generate 10 \
  --model_name="text-davinci-003" \
"""
import time
import json
import os
import random
import re

import numpy as np
import tqdm
from rouge_chinese import Rouge
import utils

import argparse
import jieba


def rouge_l_f1_score(tokens1, tokens2):
    """Compute ROUGE-L F1 score."""
    scores = Rouge().get_scores(tokens1, tokens2)
    return scores[0]['rouge-l']['f']


def encode_prompt_zh(prompt_instructions):
    """Encode multiple prompt instructions into a single string."""
    prompt = open("./data/prompt_zh.txt").read() + "\n"

    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input,
         output) = task_dict["instruction"], task_dict["input"], task_dict["output"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        input = "<noinput>" if input.lower() == "" else input
        prompt += f"###\n"
        prompt += f"{idx + 1}. instruction: {instruction}\n"
        prompt += f"{idx + 1}. input:\n{input}\n"
        prompt += f"{idx + 1}. output:\n{output}\n"
    prompt += f"###\n"
    prompt += f"{idx + 2}. instruction:"
    return prompt


def post_process_gpt3_response_zh(num_prompt_instructions, response):
    print(response)
    if response is None:
        return []
    raw_instructions = f"{num_prompt_instructions+1}. instruction:" + \
        response["text"]
    raw_instructions = re.split("###", raw_instructions)
    instructions = []
    for idx, inst in enumerate(raw_instructions):
        # if the decoding stops due to length, the last example is likely truncated so we discard it
        if idx == len(raw_instructions) - 1 and response["finish_reason"] == "length":
            print(f"Discarded due to truncation: {inst}")
            continue

        idx += num_prompt_instructions + 1
        splitted_data = re.split(
            f"{idx}\.\s*(instruction|input|output):", inst)

        if len(splitted_data) != 7:
            print(f"Incomplete data: {splitted_data}")
            continue

        else:
            inst = splitted_data[2].strip()
            input = splitted_data[4].strip()
            input = "" if input.lower() == "<noinput>" else input
            output = splitted_data[6].strip()

        # filter based on keywords that are not suitable for language models.
        blacklist = [
            "圖片",
            "圖表",
            "圖形",
            "檔案",
            "地圖",
            "繪圖",
            "繪制",
            "前往",
            "視頻",
            "影片",
            "音頻",
            "音樂",
            "流程圖",
            "示意圖",
        ]
        blacklist += []

        if any(find_word_in_chinese_string(word, inst) for word in blacklist):
            print(f"Filtered: {inst}")
            continue

        instructions.append(
            {"instruction": inst, "input": input, "output": output})
    return instructions


def find_word_in_chinese_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def generate_instruction_following_data_zh(
    output_dir="./",
    seed_tasks_path="data/seed_tasks_zh.jsonl",
    num_instructions_to_generate=1,
    model_name="text-davinci-003",
    num_prompt_instructions=1,
    request_batch_size=1,
    temperature=0,
    top_p=0,
    num_cpus=16,
):

    print("Loading seed tasks...")
    seed_tasks = [json.loads(l) for l in open(
        seed_tasks_path, "r", encoding="utf-8")]

    seed_instruction_data = []
    for t in seed_tasks:
        instance = t["instances"][0]
        if "output" in instance:
            seed_instruction_data.append(
                {
                    "instruction": t["instruction"],
                    "input": instance["input"],
                    "output": instance["output"],
                }
            )

    print(
        f"Loaded {len(seed_instruction_data)} human-written seed instructions")

    os.makedirs(output_dir, exist_ok=True)
    request_idx = 0
    # load the LM-generated instructions
    machine_instruction_data = []
    if os.path.exists(os.path.join(output_dir, "regen.json")):
        machine_instruction_data = utils.jload(
            os.path.join(output_dir, "regen.json"))
        print(
            f"Loaded {len(machine_instruction_data)} machine-generated instructions")

    # similarities = {}

    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
    if machine_instruction_data:
        progress_bar.update(len(machine_instruction_data))

    # first we tokenize all the seed instructions and generated machine instructions
    all_instructions = [d["instruction"] for d in seed_instruction_data] + [
        d["instruction"] for d in machine_instruction_data
    ]
    all_instruction_tokens = [' '.join(jieba.cut(inst))
                              for inst in all_instructions]

    while len(machine_instruction_data) < num_instructions_to_generate:
        request_idx += 1

        batch_inputs = []
        for _ in range(request_batch_size):
            # only sampling from the seed tasks
            prompt_instructions = random.sample(
                seed_instruction_data, num_prompt_instructions)
            prompt = encode_prompt_zh(prompt_instructions)
            batch_inputs.append(prompt)

        decoding_args = utils.OpenAIDecodingArguments(
            temperature=temperature,
            n=1,
            # hard-code to maximize the length. the requests will be automatically adjusted
            max_tokens=800,
            top_p=top_p,
            stop=["\n20", "20.", "20."],
        )

        request_start = time.time()
        results = utils.openai_completion(
            prompts=batch_inputs,
            model_name=model_name,
            batch_size=request_batch_size,
            decoding_args=decoding_args,
            # prevent the <|endoftext|> token from being generated
            logit_bias={"50256": -100},
        )
        request_duration = time.time() - request_start

        process_start = time.time()
        instruction_data = []
        for result in results:
            new_instructions = post_process_gpt3_response_zh(
                num_prompt_instructions, result)
            print("Generated instructions:", new_instructions)
            instruction_data += new_instructions

        total = len(instruction_data)
        keep = 0
        for instruction_data_entry in instruction_data:
            # computing similarity with the pre-tokenzied instructions
            new_instruction_tokens = ' '.join(
                jieba.cut(instruction_data_entry["instruction"]))
            rouge_scores = [
                rouge_l_f1_score(new_instruction_tokens, tokens)
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
        process_duration = time.time() - process_start
        print(
            f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s")
        print(f"Generated {total} instructions, kept {keep} instructions")
        utils.jdump(machine_instruction_data,
                    os.path.join(output_dir, "regen.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate instructions using GPT-3")
    parser.add_argument(
        "task", type=str, help="The task to perform: generate_instruction_following_data")
    parser.add_argument("--output_dir", type=str, default="data/",
                        help="Output directory for the generated instructions")
    parser.add_argument("--seed_tasks_path", type=str,
                        default="data/seed_tasks.jsonl", help="Path to the seed tasks JSONL file")
    parser.add_argument("--num_instructions_to_generate", type=int,
                        default=1, help="Number of instructions to generate")
    parser.add_argument("--model_name", type=str,
                        default="text-davinci-003", help="Name of the GPT-3 model to use")
    parser.add_argument("--num_prompt_instructions", type=int,
                        default=1, help="Number of prompt instructions to use")
    parser.add_argument("--request_batch_size", type=int,
                        default=1, help="Batch size for GPT-3 requests")
    parser.add_argument("--temperature", type=float,
                        default=0.8, help="Temperature for GPT-3 sampling")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top p for GPT-3 sampling")
    parser.add_argument("--num_cpus", type=int, default=16,
                        help="Number of CPUs to use for parallel processing")

    args = parser.parse_args()

    if args.task == "generate_instruction_following_data":

        generate_instruction_following_data_zh(
            output_dir=args.output_dir,
            num_instructions_to_generate=args.num_instructions_to_generate,
            model_name=args.model_name,
            num_prompt_instructions=args.num_prompt_instructions,
            request_batch_size=args.request_batch_size,
            temperature=args.temperature,
            top_p=args.top_p,
            num_cpus=args.num_cpus,
        )
    else:
        print(f"Unknown task: {args.task}")
