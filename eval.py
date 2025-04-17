"""
Copyright (c) 2024 Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 
"""
from fastchat.model.model_adapter import get_conversation_template
from mcts_utils.llm_server import *
from agentenv.envs import WebshopEnvClient, SciworldEnvClient, TextCraftEnvClient, WebarenaEnvClient
import argparse
import os
from mcts_utils.llm_server import FuncCall

Task = os.environ["TASK"]

if Task == "webshop":
    from mcts_utils.webshop.mcts_ws import *
elif Task == "sciworld":
    from mcts_utils.sciworld.mcts_sci import *
elif Task == "textcraft":
    from mcts_utils.textcraft.mcts_tc import *
elif Task == "webarena":
    from mcts_utils.webarena.mcts_we import *

def initialize_environment(Task: str, env_server_base: str, data_len: int = 200):
    """
    Initializes the appropriate environment based on the task type.
    """
    if Task == "webshop":
        return WebshopEnvClient(env_server_base=env_server_base, data_len=data_len)
    elif Task == "sciworld":
        return SciworldEnvClient(env_server_base=env_server_base, data_len=data_len)
    elif Task == "textcraft":
        return TextCraftEnvClient(env_server_base=env_server_base, data_len=data_len)
    elif Task == "webarena":
        return WebarenaEnvClient(env_server_base=env_server_base, data_len=data_len)
    else:
        raise ValueError(f"Unknown Task: {Task}")

def setup_conversation(env):
    """
    Sets up the initial conversation for the environment.
    """
    conversation = list(env.conversation_start)
    conv = get_conversation_template('gpt-4')

    conv.append_message(conv.roles[0], conversation[0]["value"])
    conv.append_message(conv.roles[1], 'Ok.')
    observation = env.observe() if os.environ["TASK"] == "webshop" else env.info["observation"]
    conv.append_message(conv.roles[0], observation)
    return conv

def main(Task: str, model_name: str, env_server_base: str, max_steps: int):
    """
    Main execution function for handling tasks and initiating tests.
    """
    # Initialize environment
    env = initialize_environment(Task, env_server_base)

    # Load task indices
    # temp = read_json(f"test_id/{Task}_test.json")
    temp = read_json(f"mcts_utils/{Task}/{Task}_test.json")
    task_inds = [ind["item_id"].replace(f"{Task}_", "") for ind in temp]

    # Process each task index
    for idx in task_inds:
        dir_path = f"test_result/{Task}/{model_name}"
        file_path = f"{dir_path}/search_results_{idx}.json"
        
        if os.path.exists(file_path):
            print(f"{file_path} exists. Skipping.")
            continue

        env.reset(int(idx))
        conv = setup_conversation(env)
        # perform_test(FuncCallOffline(model_name=model_name), env, conv, model_name, idx, max_steps)
        perform_test(FuncCall(model_name=model_name), env, conv, model_name, idx, max_steps)

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Run MCTS tests for specified tasks.")
    parser.add_argument("--env_server_base", type=str, default="http://127.0.0.1:8000", help="Base URL for the environment server.")
    parser.add_argument("--model_name", type=str, default="gpt-4o-2024-08-06", help="Model name to be used.")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum steps allowed for a task.")
    args = parser.parse_args()

    # Load environment variables
    Task = os.environ.get("TASK")
    if not Task:
        raise ValueError("The TASK environment variable is not set.")

    # Execute main function
    main(Task, args.model_name, args.env_server_base, args.max_steps)