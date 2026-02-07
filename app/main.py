import argparse
import os
import sys
import json
import subprocess
from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")
messages = []

def read_file(file_path):
    f = open(file_path)
    text = f.read()
    f.close()
    return text

def write_file(file_path, content):
    if not file_path or content is None:
        return "Error: 'file_path' and 'content' arguments are required"
    try:
        with open(file_path, "w") as f:
            f.write(content)
        return "Write successful"
    except Exception as e:
        return f"Error writing file: {str(e)}"
    
def execute_bash(command):
    if not command:
        return "Error: 'command' argument is required"
    print(f"Executing command: {command}", file=sys.stderr)
    try:
        result = subprocess.run(
            str.split(command), check=True, text=True, capture_output=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Command failed with exit code {e.returncode}: {e.output}"

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
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
            "type": "object",
            "required": ["file_path", "content"],
            "properties": {
                "file_path": {
                "type": "string",
                "description": "The path of the file to write to"
                },
                "content": {
                "type": "string",
                "description": "The content to write to the file"
                }
            }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a shell command",
            "parameters": {
            "type": "object",
            "required": ["command"],
            "properties": {
                "command": {
                "type": "string",
                "description": "The command to execute"
                }
            }
            }
        }
    }
]

tool_functions_map = {
    "read_file": read_file,
    "write_file": write_file,
    "bash": execute_bash
}

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
        
        if not assistant_message.tool_calls:
            done = True
            continue
        
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
                                                 
    print(messages[-1]["content"])



if __name__ == "__main__":
    main()
