import copy
import dataclasses
import io
import json
import logging
import math
import os
import sys
import time
from typing import Optional, Sequence, Union, List, Dict
import backoff

import openai
import tqdm
from dotenv import load_dotenv
from openai import openai_object

StrOrOpenAIObject = Union[str, openai_object.OpenAIObject]

# Load environment variables
load_dotenv()
openai_org = os.getenv("OPENAI_ORG")
if openai_org is not None:
    openai.organization = openai_org
    logging.warning(
        f"Switching to organization: {openai_org} for OAI API key.")
openai.api_key = os.getenv("OPENAI_API_KEY")


@dataclasses.dataclass
class OpenAIDecodingArguments:
    max_tokens: int = 500
    temperature: float = 0.2
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    suffix: Optional[str] = None


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai_completion(**kwargs)


def openai_completion(
    messages: List[Dict[str, str]],
    decoding_args: OpenAIDecodingArguments,
    model_name="gpt-3.5-turbo",
    sleep_time=2,
    return_text=False,
    **decoding_kwargs,
) -> List[str]:
    completions = []

    """"https://platform.openai.com/docs/api-reference/chat"""

    # print current model
    print("Current model: ", model_name)

    while True:
        try:
            shared_kwargs = dict(
                model=model_name,
                temperature=decoding_args.temperature,
                top_p=decoding_args.top_p,
                n=decoding_args.n,
                stream=decoding_args.stream,
                stop=decoding_args.stop,
                max_tokens=decoding_args.max_tokens,
                presence_penalty=decoding_args.presence_penalty,
                frequency_penalty=decoding_args.frequency_penalty,

                **decoding_kwargs,
            )
            response = openai.ChatCompletion.create(
                messages=messages, **shared_kwargs
            )

            if return_text:
                completions = [choice['message']['content']
                               for choice in response['choices']]
            else:
                completions = response['choices']

            break
        except openai.error.OpenAIError as e:
            logging.warning(f"OpenAIError: {e}.")
            if "Please reduce your prompt" in str(e):
                decoding_args.max_tokens = int(decoding_args.max_tokens * 0.8)
                logging.warning(
                    f"Reducing target length to {decoding_args.max_tokens}, Retrying..."
                )
            else:
                logging.warning("Hit request rate limit; retrying...")
                time.sleep(sleep_time)

    # Add the optional suffix if specified
    if decoding_args.suffix:
        completions = [completion +
                       decoding_args.suffix for completion in completions]

    return completions


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default, ensure_ascii=False)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict
