import argparse
import os
import sys
import json
import subprocess
from openai import OpenAI
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory

MODEL=os.getenv("MODEL", "anthropic/claude-haiku-4.5")
API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")
messages = []

def read_file(file_path):
    if not file_path:
        return "Error: 'file_path' argument is required"
    try:
        with open(file_path, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

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

def agent_loop(client):
    """Execute the agent loop: call LLM, handle tool calls, return final response.
    
    Args:
        client: OpenAI client instance
    
    Returns:
        str: Final assistant response content, or None if error occurred
    """
    done = False
    
    while not done:
        chat = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools
        )
        
        if not chat.choices or len(chat.choices) == 0:
            return None
        
        assistant_message = chat.choices[0].message
        messages.append({
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": assistant_message.tool_calls,
        })
        
        if not assistant_message.tool_calls:
            done = True
            return assistant_message.content
        
        # Execute tool calls
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            argumentsJson = tool_call.function.arguments
            
            if function_name in tool_functions_map:
                tool_function = tool_functions_map[function_name]
                try:
                    arguments = json.loads(argumentsJson)
                    result = tool_function(**arguments)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
                except Exception as e:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"Error calling tool function '{function_name}': {str(e)}"
                    })
                    print(f"Error calling tool function '{function_name}': {e}", file=sys.stderr)
    
    return None

def run_repl():
    """Interactive REPL mode for continuous conversations"""
    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    
    session = PromptSession(history=InMemoryHistory(), multiline=False)
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    print("AI Coding Assistant (type /help for commands, /exit to quit)")
    
    while True:
        try:
            user_input = session.prompt(">>> ")
            
            if not user_input.strip():
                continue
            
            # Handle special commands
            if user_input == "/exit":
                print("Goodbye!")
                break
            elif user_input == "/clear":
                # Clear conversation but keep system messages
                messages.clear()
                print("Conversation cleared.")
                continue
            elif user_input == "/help":
                print("Commands:")
                print("  /exit  - Exit the REPL")
                print("  /clear - Clear conversation history")
                print("  /help  - Show this help message")
                continue
            
            # Add user message
            messages.append({"role": "user", "content": user_input})
            
            # Run agent loop
            try:
                response = agent_loop(client)
                if response is None:
                    print("Error: No response from API", file=sys.stderr)
                    # Remove the last user message since we failed to process it
                    if messages and messages[-1]["role"] == "user":
                        messages.pop()
                elif response:
                    print(response)
            except Exception as e:
                print(f"API Error: {str(e)}")
                # Remove the last user message since we failed to process it
                if messages and messages[-1]["role"] == "user":
                    messages.pop()
        
        except KeyboardInterrupt:
            print("\n(Use /exit to quit)")
            continue
        except EOFError:
            print("\nGoodbye!")
            break

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=False)
    p.add_argument("-i", "--interactive", action="store_true", help="Start interactive REPL mode")
    args = p.parse_args()

    if args.interactive:
        run_repl()
        return

    if not args.p:
        print("Error: Either -p or -i/--interactive is required")
        sys.exit(1)

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    message = {
        "role": "user",
        "content": args.p
    }
    messages.append(message)
    
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    response = agent_loop(client)
    if response is None:
        raise RuntimeError("no choices in response")
    
    print(response)

if __name__ == "__main__":
    main()
