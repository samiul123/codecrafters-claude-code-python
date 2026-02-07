import argparse
import os
import sys
import json
from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")

def _debug(message: str) -> None:
    if os.getenv("DEBUG") == "1":
        print(f"debug: {message}", file=sys.stderr)

def _get_attr(obj, name: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)

def _normalize_message_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                    continue
            parts.append(str(item))
        return "".join(parts)
    return str(content)

def _execute_read_tool(arguments_json: str) -> None:
    try:
        arguments = json.loads(arguments_json)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"invalid tool arguments JSON: {e}") from e

    file_path = arguments.get("file_path")
    if not isinstance(file_path, str) or not file_path:
        raise RuntimeError("Read tool requires a non-empty file_path")

    with open(file_path, "rb") as f:
        sys.stdout.buffer.write(f.read())

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    chat = client.chat.completions.create(
        model="anthropic/claude-haiku-4.5",
        messages=[{"role": "user", "content": args.p}],
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read and return the contents of a file",
                    "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                        "type": "string",
                        "description": "The path to the file to read"
                        }
                    },
                    "required": ["file_path"]
                    }
                }
            }
        ]
    )

    if not chat.choices or len(chat.choices) == 0:
        raise RuntimeError("no choices in response")

    # You can use print statements as follows for debugging, they'll be visible when running tests.
    # print("Logs from your program will appear here!", file=sys.stderr)

    # TODO: Uncomment the following line to pass the first stage
    # print(chat.choices[0].message.content)
    
    message = chat.choices[0].message
    tool_calls = _get_attr(message, "tool_calls")
    if tool_calls:
        tool_call = tool_calls[0]
        function = _get_attr(tool_call, "function")
        name = _get_attr(function, "name")
        arguments = _get_attr(function, "arguments")

        _debug(f"tool call: {name}")

        if name != "read_file":
            raise RuntimeError(f"unsupported tool: {name}")
        if not isinstance(arguments, str):
            raise RuntimeError("tool arguments must be a JSON string")

        _execute_read_tool(arguments)
        return

    sys.stdout.write(_normalize_message_content(_get_attr(message, "content")))



if __name__ == "__main__":
    main()
