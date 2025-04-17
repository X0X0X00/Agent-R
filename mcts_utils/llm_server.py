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
import os
import json
import csv
import os
import openai
import tiktoken
import asyncio
import jsonlines
from transformers import AutoTokenizer
from mcts_utils.sciworld.eval_utils_sw import findValidActionNew
from utils.logger import logger


class FuncCallOffline:
    def __init__(self, model_name=None):
        from vllm import LLM, SamplingParams
        self.model_name = model_name
        self.llm = LLM(model=os.environ["MODEL_DIR"], dtype="half")
        if "TEMP" in os.environ:
            print(f'当前 TEMP 值: {os.environ["TEMP"]}')
        else:
            os.environ["TEMP"] = "1"
            print("TEMP 不存在，已设置为 1")
        self.sampling_params = SamplingParams(temperature=float(os.environ["TEMP"]), max_tokens=500, stop=["<|eot_id|>"])
        tokenizer = AutoTokenizer.from_pretrained(os.environ["MODEL_DIR"])
        self.encoding = tokenizer

    def llm_func(self, messages, model_name):
        logger.debug(f"FuncCallOffline.llm_func - Messages: {messages}")
        outputs = self.llm.chat(messages, sampling_params=self.sampling_params)
        text = outputs[0].outputs[0].text.strip()
        logger.debug(f"FuncCallOffline.llm_func - Cleaned Result: {text}")
        return text
    
    async def llm_func_async(self, messages, model_name):
        """Asynchronous version of llm_func"""
        logger.debug(f"FuncCallOffline.llm_func_async - Messages: {messages}")
        # Use asyncio.to_thread to run the synchronous method in a thread pool
        text = await asyncio.to_thread(self.llm_func, messages, model_name)
        return text
    

class FuncCall:
    def __init__(self, model_name=None):
        self.model_name = model_name
        token_model = 'gpt-4o-mini'
        # self.encoding = tiktoken.encoding_for_model(token_model)
        self.encoding = tiktoken.get_encoding("cl100k_base")

        # Create a semaphore to limit concurrent API calls
        self.api_semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent API calls

    def message_construction(self, prompt, model_name=""):
        if model_name != 'gemini':
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = [{"role": "user", "parts": [prompt]}]
        return messages

    def llm_func(self, messages, model_name):
        logger.debug(f"FuncCall.llm_func - Messages: {messages}")
        ind = 0
        while True:
            try:
                if "gpt" in model_name or True:
                    client = openai.OpenAI(
                       base_url = os.environ["OPENAI_API_BASE"],
                        api_key = os.environ["OPENAI_API_KEY"],
                    )
                    temperature = float(os.environ["TEMP"])


                    # result = client.chat.completions.create(
                    #     model =model_name,
                    #     messages=messages,
                    #     temperature=temperature,
                    #     stop=None)
                    # clean_result = result.choices[0].message.content
                    print("[LLM DEBUG] Sending prompt to model:", messages)
                    print("[LLM DEBUG] Model name:", model_name)

                    try:
                        result = client.chat.completions.create(
                            model = model_name,
                            messages = messages,
                            temperature = temperature,
                            stop=None,
                            max_tokens=512
                        )
                        print("[LLM] result:", result)
                        clean_result = result.choices[0].message.content
                    except Exception as e:
                        print("[LLM ERROR] Failed:", e)
                        return ""



                    
                logger.debug(f"FuncCall.llm_func - Cleaned Result: {clean_result}")
                return clean_result
            except Exception as e:
                logger.error(f"FuncCall.llm_func - Error: {e}")
                if ind > 100:
                    return -1
                ind += 1
                continue


def perform_test(calling, env, conv, model_name, idx, max_steps):
    Task = os.environ["TASK"]
    model_type = os.environ["MODEL_TYPE"]
    dir_path = f"test_result/{Task}/{model_name}_{model_type}"
    file_path = f"{dir_path}/search_results_{idx}.json"

    os.makedirs(dir_path, exist_ok=True)
    done = False
    max_steps = max_steps
    current_step = 0
    new_env_score = 0
    current_recent_actions = []
    while not done:
        if current_step >= max_steps:
            done = True
            continue

        max_len = 7000
        if "MAX_TOKEN_LENGTH" in os.environ:
            max_len = int(os.environ["MAX_TOKEN_LENGTH"])
        while len(calling.encoding.encode(str(conv))) > max_len - 60:
            del conv.messages[4:6]
        current_step += 1
        prompt = conv.to_openai_api_messages()
        agent_response = calling.llm_func(prompt, model_name)
        new_action = agent_response.split('Action:')[-1].strip()
        if Task == "sciworld":
            new_action = findValidActionNew([new_action], env, env.get_look_around(), current_recent_actions)
        step_output = env.step(new_action)
        env_state, env_reward, env_done = (
                        step_output.state,
                        step_output.reward,
                        step_output.done,
                    )
        current_obs = env.observe()
        done = env_done
        new_env_score = env_reward
        conv.append_message(conv.roles[1], None)
        conv.update_last_message(agent_response)
        conv.append_message(conv.roles[0], current_obs)

        if new_env_score < 0:
            done = True
            new_env_score = 0
        print(new_action)
        print(env_state)
        print(env_reward)
        current_recent_actions.append(f'({new_action}, {current_obs})')


    final_result = {
        "task_id": idx,
        "env_score": new_env_score,
        "model_name": model_name,
        "step_num": current_step,
        "state": conv.to_openai_api_messages(),
        }
    
    save_json(final_result, file_path)

def perform_test_revise(calling, env, conv, model_name, idx, max_steps, content_ls):
    Task = os.environ["TASK"]
    model_type = os.environ["MODEL_TYPE"]
    dir_path = f"revise_result/{Task}/{model_name}_{model_type}"
    file_path = f"{dir_path}/search_results_{idx}.json"
    os.makedirs(dir_path, exist_ok=True)
    done = False
    max_steps = max_steps
    current_step = 0
    new_env_score = 0
    current_recent_actions = []
    for content in content_ls:
        agent_response = content
        new_action = agent_response.split('Action:')[-1].strip()
        if Task == "sciworld":
            new_action = findValidActionNew([new_action], env, env.get_look_around(), current_recent_actions)
        step_output = env.step(new_action)
        env_state, env_reward, env_done = (
                        step_output.state,
                        step_output.reward,
                        step_output.done,
                    )
        current_obs = env.observe()
        done = env_done
        new_env_score = env_reward
        conv.append_message(conv.roles[1], None)
        conv.update_last_message(agent_response)
        conv.append_message(conv.roles[0], current_obs)

    from copy import deepcopy
    history = deepcopy(conv)
    max_len = 7000
    if "MAX_TOKEN_LENGTH" in os.environ:
        max_len = int(os.environ["MAX_TOKEN_LENGTH"])
    while not done:
        if current_step >= max_steps:
            done = True
            continue
        while len(calling.encoding.encode(str(conv))) > max_len - 60:
            del conv.messages[4:6]
        current_step += 1
        prompt = conv.to_openai_api_messages()
        agent_response = calling.llm_func(prompt, model_name)
        new_action = agent_response.split('Action:')[-1].strip()
        if Task == "sciworld":
            new_action = findValidActionNew([new_action], env, env.get_look_around(), current_recent_actions)
        step_output = env.step(new_action)
        env_state, env_reward, env_done = (
                        step_output.state,
                        step_output.reward,
                        step_output.done,
                    )
        current_obs = env.observe()
        done = env_done
        new_env_score = env_reward
        conv.append_message(conv.roles[1], None)
        conv.update_last_message(agent_response)
        conv.append_message(conv.roles[0], current_obs)

        history.append_message(conv.roles[1], None)
        history.update_last_message(agent_response)
        history.append_message(conv.roles[0], current_obs)

        if new_env_score < 0:
            done = True
            new_env_score = 0
        print(new_action)
        print(env_state)
        print(env_reward)
        current_recent_actions.append(f'({new_action}, {current_obs})')


    final_result = {
        "task_id": idx,
        "env_score": new_env_score,
        "model_name": model_name,
        "step_num": current_step,
        "state": history.to_openai_api_messages(),
        }
    
    save_json(final_result, file_path)

def get_last_processed_index(progress_file):
    """Retrieve the last processed index from the progress file."""
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            last_index = f.read().strip()
            return int(last_index) if last_index else 0
    else:
        return 0


def update_progress(progress_file, index):
    """Update the last processed index in the progress file."""
    with open(progress_file, 'w', encoding='utf-8') as f:
        f.write(str(index))


def read_jsonline(address):
    not_mark = []
    with open(address, 'r', encoding="utf-8") as f:
        for jsonstr in f.readlines():
            jsonstr = json.loads(jsonstr)
            not_mark.append(jsonstr)
    return not_mark


def read_json(address):
    with open(address, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
    return json_data

import os

def create_file_if_not_exists(file_path):
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            pass  # 创建空文件

def read_jsonl(address):
    not_mark = []
    with open(address, "r", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            not_mark.append(item)
    return not_mark


def read_csv(address):
    dataset = []
    with open(address, encoding='utf-8-sig') as f:
        for row in csv.reader(f, skipinitialspace=True):
            dataset.append(row)
    return dataset


def read_tsv(address):
    dataset = []
    with open(address, encoding='utf-8-sig') as f:
        tsvreader = csv.reader(f, delimiter='\t')
        for row in tsvreader:
            dataset.append(row)
    return dataset


def read_txt(address, sep):
    dataset = []
    with open(address, 'r', encoding="utf-8") as f:
        for data in f.readlines():
            data = data.replace('\n', '').split(sep)
            dataset.append(data)
    return dataset


def save_jsonline(ls, address):
    for item in ls:
        with open(address, 'a+', encoding='utf-8') as f:
            line = json.dumps(item, ensure_ascii=False)
            f.write(line + '\n')


def save_json(ls, address):
    #json_str = json.dumps(ls, indent=4)
    with open(address, 'w', encoding='utf-8') as json_file:
        json.dump(ls, json_file, ensure_ascii=False, indent=4)


def sort_dic(dic):
    dic = sorted(dic.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    return dic


def expand_dataset(dataset, expand_time):
    final_dataset = []
    for item in dataset:
        for i in range(expand_time):
            final_dataset.append(item)
    return final_dataset

def rewrite(dataset, output_path):
    final_dataset = []
    for data in dataset:
        conversation = []
        revise_log = data["revise_log"]
        temp = {
            "system": revise_log[0]["content"],
            "input": revise_log[1]["content"],
            "output": revise_log[2]["content"],
        }
        conversation.append(temp)
        i = 3
        while i+1 < len(revise_log):
            temp = {
                "input": revise_log[i]["content"],
                "output": revise_log[i+1]["content"],
            }
            i += 2
            conversation.append(temp)
        final_dataset.append({"conversation": conversation})

    save_json(final_dataset, output_path)

def write_to_jsonl(output_file, data):
    """
    Appends data to a JSONL file.

    Args:
        output_file: Path to the JSONL file.
        data: Data to append as a dictionary.
    """
    with open(output_file, 'a+', encoding='utf-8') as jsonl_file:
        jsonl_file.write(json.dumps(data, ensure_ascii=False) + '\n')

revision_thoughts = [
    "I realize my approach was flawed. I need to revise it.",
    "I took the wrong steps. I need to identify the right path.",
    "My actions were incorrect. I must adjust my strategy.",
    "I see an error in my actions. I need to fix it.",
    "I misunderstood the situation. Time to reassess.",
    "My decision was wrong. I should reevaluate.",
    "I went off course. I need to steer back on track.",
    "I recognize my mistake. Let’s find a better solution.",
    "My judgment was incorrect. I need to rethink it.",
    "I made an error. I must determine how to correct it.",
    "I acted hastily. I need to reconsider my choices.",
    "I misjudged the scenario. Time to reflect and adjust.",
    "My initial steps were wrong. I need a new approach.",
    "I realize I chose poorly. I must change direction.",
    "I overlooked something important. I need to address it.",
    "I miscalculated. It’s time to figure out a better way.",
    "I made a poor decision. I need to set things right.",
    "I recognize my failure. I need to learn and move forward.",
    "I didn’t think this through. I must reevaluate.",
    "I strayed from the goal. I need to realign my efforts."
]

prompt_template = """
You are a good verifier. You will be given a log that records an agent interacting with an environment to solve a science task. The format of the log is:
```
Action: #Action
Observation: #Observation
```

You need to verify whether the current action is good, bad, or uncertain in the log. 
- A **good** action is greatly helpful to solve the task.
- A **bad** action is greatly harmful to solve the task.
- An **uncertain** action is one that is neither good nor bad, or you cannot judge based on the current information.

Log:
Task Description: {task_description}
{action_obs_prompt}
Current_Action: {action}
Current_Observation: {observation}

You must give reasons first and then provide the response in the format: Judgement: <Good or Bad or Uncertain>
""".strip()