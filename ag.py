#!/usr/bin/env python3

import argparse
import sys
import os
from openai import OpenAI
import easyocr
import warnings

from rich.console import Console
from rich.markdown import Markdown

def display_markdown_with_rich(markdown_string):
    console = Console()
    md = Markdown(markdown_string)
    console.print(md)

def ocr(image_input):
    reader = easyocr.Reader(['ch_sim', 'en'])
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                message=".*'pin_memory' argument is set as true but not supported on MPS now.*",
                                category=UserWarning)

            results = reader.readtext(image_input)
            texts = [result[1] for result in results]
            return '\n'.join(texts).strip()
    except Exception as e:
        print(f"OCR 错误: {str(e)}", file=sys.stderr)
        return None

def get_client():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("Failed to find API key.", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

def call_deepseek(messages, model, stream):
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=stream
    )
    return response

def print_response(response, model, stream):
    print(f"Model({model}):\n")
    full_response= ''
    if stream:
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content_chunk = chunk.choices[0].delta.content
                full_response += content_chunk
        display_markdown_with_rich(full_response)
        columns = os.get_terminal_size().columns
        print('=' * columns)
    else:
        print(f"    {response.choices[0].message.content}")
        columns = os.get_terminal_size().columns
        print('=' * columns)
        full_response = response.choices[0].message.content
    return full_response


def query_mode(query, contents, image=False, deep=False, stream=True):
    if contents != "":
        if image:
            prompt = f"This is ocr result of the image:\n\n{contents}\n\nQuestion:{query}"
        else:
            prompt = f"Please answer the question based on this text:\n\n{contents}\n\nQuestion:{query}"
    else:
        prompt = query

    professional_prompt="""
    Please answer my question with your output formatted in strict Markdown syntax, ensuring 
    it's directly compatible with Glow for display. Keep concise. Determine your language 
    according to my question. Please do not use bold or italian.
    Try not use titles (e.g., #).
    My question is:
    """
    original_prompt = prompt
    prompt = (f"{professional_prompt}{original_prompt}")

    model = "deepseek-chat"
    if deep:
        model = "deepseek-reasoner"

    messages = [
        {"role": "user", "content": prompt}
    ]
    response = call_deepseek(messages, model, stream)
    print_response(response, model, stream)

def chat_mode(deep=False, stream=True):
    system_prompt = (
        "Please give precise and crispy answers. Faster is better."
    )

    model = "deepseek-chat"
    if deep:
        model = "deepseek-reasoner"

    messages = [{"role": "system", "content": system_prompt}]
    print("Entering chat mode. Type 'exit', 'quit', or 'q' to exit.")
    print("--------------------------------------")
    while True:
        try:
            user_input = input("\nUser\n").strip()

            if user_input.lower() in ['exit', 'quit', 'q']:
                break

            messages.append({"role": "user", "content": user_input})
            response = call_deepseek(messages, model, stream)

            reply = print_response(response, model, stream)
            messages.append({"role": "assistant", "content": reply})
        except KeyboardInterrupt:
            print("\nType 'exit', 'quit', or 'q' to exit.")
        except Exception as e:
            print(f"Error: {str(e)}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(
        description="(LLM) AGent for work.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-q', '--query',  nargs='?', const='',
                       help='Query content from standard input.')
    parser.add_argument('-i', '--image', action='store_true',
                       help='Process image from standard input (perform OCR).')
    parser.add_argument('-d', '--deep', action='store_true',
                       help='Use deepseek-reasoner model with chain-of-thought reasoning.')
    parser.add_argument('-c', '--chat', action='store_true',
                       help='Enter interactive chat mode.')
    parser.add_argument('-s', '--stream',
                       help='Enable streaming response output (default: enabled).', default=True)

    args = parser.parse_args()

    if args.chat:
        chat_mode(args.deep, args.stream)
        return

    contents = ''
    if not sys.stdin.isatty():
        if args.image:
            contents = sys.stdin.buffer.read()
            contents = ocr(contents)
            # print(contents)
        else:
            contents = sys.stdin.read().strip()

    if args.query:
        query = args.query
        if not query:
            print("Error: Please provide a query.", file=sys.stderr)
            sys.exit(1)
        query_mode(query, contents, args.image, args.deep, args.stream)


if __name__ == "__main__":
    main()
