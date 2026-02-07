import argparse
import os
import sys
import json
from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")
messages = []

def _debug(message: str) -> None:
    if os.getenv("DEBUG") == "1":
        print(f"debug: {message}", file=sys.stderr)

def _get_attr(obj, name: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)

def read_file(file_path):
    f = open(file_path)
    text = f.read()
    f.close()
    return text

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

tool_functions_map = {
    "read_file": read_file
}

# def _normalize_message_content(content) -> str:
#     if content is None:
#         return ""
#     if isinstance(content, str):
#         return content
#     if isinstance(content, list):
#         parts: list[str] = []
#         for item in content:
#             if isinstance(item, str):
#                 parts.append(item)
#                 continue
#             if isinstance(item, dict):
#                 text = item.get("text")
#                 if isinstance(text, str):
#                     parts.append(text)
#                     continue
#             parts.append(str(item))
#         return "".join(parts)
#     return str(content)

# def _execute_read_tool(arguments_json: str) -> None:
#     try:
#         arguments = json.loads(arguments_json)
#     except json.JSONDecodeError as e:
#         raise RuntimeError(f"invalid tool arguments JSON: {e}") from e

#     file_path = arguments.get("file_path")
#     if not isinstance(file_path, str) or not file_path:
#         raise RuntimeError("Read tool requires a non-empty file_path")

#     with open(file_path, "rb") as f:
#         sys.stdout.buffer.write(f.read())



def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    message = {
        "role": "user",
        "content": args.p
    }
    messages.append(message)
    
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    done = False

    while not done:
        chat = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=messages,
            tools = tools
        )

        if not chat.choices or len(chat.choices) == 0:
            raise RuntimeError("no choices in response")
        
        assistant_message = chat.choices[0].message
        messages.append(
            {
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": assistant_message.tool_calls,
            }
        )
        
        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                argumentsJson = tool_call.function.arguments
                
                # call the tool function and provide appropriate arguments
                if function_name in tool_functions_map:
                    tool_function = tool_functions_map[function_name]
                    try:
                        arguments = json.loads(argumentsJson)
                        result = tool_function(**arguments)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": result
                            })
                        
                    except Exception as e:
                        print(f"Error calling tool function '{function_name}': {e}", file=sys.stderr)
                                         
        else:
            done = True
    
    print(messages[-1]["content"])



if __name__ == "__main__":
    main()
