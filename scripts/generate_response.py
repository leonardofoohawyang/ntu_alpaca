import dataclasses
import datetime
import json
import logging
import os
import time
from typing import Dict, List, Optional, Sequence

import openai
import pandas as pd
from dotenv import load_dotenv

import utils


def generate_response(prompt, model="gpt-4", temperature=0.1):

    sys_cmd = "你是一個樂於助人的助手。"
    user_cmd = f"請以繁體中文回答: {prompt}。"

    messages = [{"role": "system", "content": sys_cmd},
                {"role": "user", "content": user_cmd}]

    decoding_args = utils.OpenAIDecodingArguments(
        temperature=temperature,
        max_tokens=3072,
        top_p=1.0,
        n=1,
        stream=False,
        stop=["\n"],
        presence_penalty=0.0,
        frequency_penalty=0.0,
    )

    response = utils.completions_with_backoff(
        messages=messages,
        model_name=model,
        decoding_args=decoding_args,
    )

    print(response)

    return response[0]["message"]["content"]


def dump_response(input_file='data/instructions_0427_2106.json', output_file='data/pairs_0427_gpt4.json', model="gpt-3.5-turbo", temperature=0.1):
    # Load instruction and input data from JSON file
    data = pd.read_json(input_file)

    # Write the opening bracket for the JSON array
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('[')

    # Iterate through the rows of the DataFrame and generate responses
    for index, row in data.iterrows():
        instruction, input_data = row['instruction'], row['input']
        prompt = f"{instruction} {input_data}"
        response = generate_response(
            prompt, model=model, temperature=temperature)

        # Dump each received response
        print(
            f"Instruction: {instruction}\nInput: {input_data}\nResponse: {response}\n")

        # Create a dictionary for the current record
        record = {'instruction': instruction,
                  'input': input_data, 'response': response}

        # Append the current record to the JSON file
        with open(output_file, 'a', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False)
            if index < len(data) - 1:
                f.write(',\n')

    # Write the closing bracket for the JSON array
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(']')


def load_instructions(input_file="data/regen.json"):
    # Read the JSON file using pandas
    data = pd.read_json(input_file)

    # Select only the 'instruction' and 'input' columns
    data = data[['instruction', 'input']]

    # Generate a filename based on the current date and number of instructions
    timestamp = datetime.datetime.now().strftime('%y%m%d')
    num_instructions = len(data)
    output_file = f"data/instructions_{timestamp}_{num_instructions}.json"

    # Save the resulting DataFrame as a JSON file with traditional Chinese characters preserved
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(data.to_json(orient='records', force_ascii=False))

    # Display success message
    print(f"Successfully dumped {num_instructions} data into {output_file}.")


if __name__ == "__main__":
    # load_instructions()
    dump_response(model="gpt-3.5-turbo",
                  input_file="data/instructions_230427_4684.json")
