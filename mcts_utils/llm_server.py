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
import re
import time
import openai
import tiktoken
import numpy as np
import logging
import jsonlines
from datetime import datetime
from transformers import AutoTokenizer
from browser_env.actions import (
    create_id_based_action,
)
# 设置详细的日志记录
def setup_detailed_logger():
    """设置详细的日志记录器"""
    logger = logging.getLogger('LLM_API_Logger')
    logger.setLevel(logging.DEBUG)
    
    # 如果已经有handler了，就不重复添加
    if logger.handlers:
        return logger
    
    # 创建文件handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"llm_api_detailed_{timestamp}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # 创建控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建详细的formatter
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    file_handler.setFormatter(detailed_formatter)
    console_handler.setFormatter(simple_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"=== LLM API Logger Started - Log file: {log_file} ===")
    return logger

# 全局logger
api_logger = setup_detailed_logger()

def log_environment_info():
    """记录环境变量信息"""
    api_logger.info("=== Environment Variables ===")
    env_vars = [
        "OPENAI_API_KEY", "OPENAI_API_BASE", "OPENAI_API_TYPE", 
        "OPENAI_API_VERSION", "MODEL_DIR", "TASK", "MODEL_TYPE",
        "MAX_TOKEN_LENGTH", "TEMP"
    ]
    
    for var in env_vars:
        value = os.environ.get(var, "NOT SET")
        if "KEY" in var and value != "NOT SET":
            # 只显示API密钥的前10个字符
            display_value = f"{value[:10]}...***"
        else:
            display_value = value
        api_logger.info(f"{var}: {display_value}")
    
    api_logger.info("=" * 40)

def log_request_details(messages, model_name, attempt_num=1):
    """记录请求详情"""
    api_logger.info(f"=== API Request Details (Attempt {attempt_num}) ===")
    api_logger.info(f"Model Name: {model_name}")
    api_logger.info(f"Message Count: {len(messages)}")
    
    total_chars = 0
    # 记录每条消息的详情
    for i, msg in enumerate(messages):
        api_logger.info(f"Message {i+1}:")
        api_logger.info(f"  Role: {msg.get('role', 'UNKNOWN')}")
        
        # 记录内容（截断长内容）
        if 'content' in msg:
            content = str(msg['content'])
            total_chars += len(content)
            if len(content) > 500:
                api_logger.info(f"  Content: {content[:500]}... (truncated, total length: {len(content)})")
            else:
                api_logger.info(f"  Content: {content}")
        elif 'parts' in msg:
            parts = str(msg['parts'])
            total_chars += len(parts)
            if len(parts) > 500:
                api_logger.info(f"  Parts: {parts[:500]}... (truncated, total length: {len(parts)})")
            else:
                api_logger.info(f"  Parts: {parts}")
        else:
            msg_str = json.dumps(msg, ensure_ascii=False, indent=2)
            total_chars += len(msg_str)
            api_logger.info(f"  Full Message: {msg_str}")
    
    api_logger.info(f"Total request size: {total_chars} characters")
    
    # 警告如果内容过大
    if total_chars > 50000:
        api_logger.warning(f"WARNING: Request size ({total_chars} chars) is very large and may cause API errors!")
    
    # 记录完整的JSON请求（用于调试）
    try:
        full_request = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 500
        }
        request_str = json.dumps(full_request, ensure_ascii=False, indent=2)
        api_logger.debug(f"Full Request JSON size: {len(request_str)} characters")
        
        # 只记录前1000字符的请求内容到debug日志
        if len(request_str) > 1000:
            api_logger.debug(f"Full Request JSON (truncated): {request_str[:1000]}...")
        else:
            api_logger.debug(f"Full Request JSON: {request_str}")
    except Exception as e:
        api_logger.error(f"Failed to serialize full request: {e}")
    
    api_logger.info("=" * 40)

def log_response_details(response, success=True, error=None):
    """记录响应详情"""
    if success:
        api_logger.info("=== API Response Details ===")
        api_logger.info("Status: SUCCESS")
        api_logger.info(f"Response: {response}")
        api_logger.info(f"Response Length: {len(str(response))}")
    else:
        api_logger.error("=== API Error Details ===")
        api_logger.error(f"Status: FAILED")
        api_logger.error(f"Error: {error}")
        api_logger.error(f"Error Type: {type(error).__name__}")
    api_logger.info("=" * 40)

# 导入 WebArena 相关的动作验证函数（如果需要）
try:
    from mcts_utils.sciworld.eval_utils_sw import findValidActionNew
except ImportError:
    def findValidActionNew(actions, env, look_around, recent_actions):
        return actions[0] if actions else ""

# Add this helper function to convert numpy arrays to JSON-serializable format
def make_json_serializable(obj):
    """Convert numpy arrays and other non-serializable objects to JSON-compatible format"""
    if obj is None:
        return None
    elif isinstance(obj, (np.ndarray, np.generic)):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        # 特殊处理：如果是message content格式 {'text': '...'}, 提取text值
        if len(obj) == 1 and 'text' in obj:
            api_logger.info(f"Converting message content dict to string: {str(obj)[:100]}...")
            return str(obj['text'])
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, str):
        return obj
    elif isinstance(obj, (int, float)):
        return obj
    else:
        # For other types, try to convert to string if they can't be serialized
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            api_logger.warning(f"Converting non-serializable object of type {type(obj)} to string: {str(obj)[:100]}...")
            return str(obj)


class FuncCallOffline:
    def __init__(self, model_name=None):
        api_logger.info("=== Initializing FuncCallOffline ===")
        log_environment_info()
        
        from vllm import LLM, SamplingParams
        self.model_name = model_name
        api_logger.info(f"Loading model from: {os.environ.get('MODEL_DIR', 'NOT SET')}")
        
        self.llm = LLM(model=os.environ["MODEL_DIR"], dtype="half")
        if "TEMP" in os.environ:
            print(f'当前 TEMP 值: {os.environ["TEMP"]}')
            api_logger.info(f'当前 TEMP 值: {os.environ["TEMP"]}')
        else:
            os.environ["TEMP"] = "1"
            print("TEMP 不存在，已设置为 1")
            api_logger.info("TEMP 不存在，已设置为 1")
        
        self.sampling_params = SamplingParams(temperature=float(os.environ["TEMP"]), max_tokens=500, stop=["<|eot_id|>"])
        api_logger.info(f"Sampling params: {self.sampling_params}")
        
        tokenizer = AutoTokenizer.from_pretrained(os.environ["MODEL_DIR"])
        self.encoding = tokenizer
        api_logger.info("FuncCallOffline initialized successfully")

    def llm_func(self, messages, model_name):
        api_logger.info("=== FuncCallOffline.llm_func Called ===")
        
        # Convert any numpy arrays in messages to JSON-serializable format
        try:
            original_messages_str = str(messages)
            messages = make_json_serializable(messages)
            new_messages_str = str(messages)
            
            if original_messages_str != new_messages_str:
                api_logger.info("Messages contained numpy arrays - converted to JSON-serializable format")
        except Exception as e:
            api_logger.warning(f"Could not compare messages for numpy detection: {e}")
            messages = make_json_serializable(messages)
        
        log_request_details(messages, model_name)
        
        try:
            api_logger.info("Calling self.llm.chat...")
            outputs = self.llm.chat(messages, sampling_params=self.sampling_params)
            text = outputs[0].outputs[0].text.strip()
            
            log_response_details(text, success=True)
            return text
            
        except Exception as e:
            log_response_details(None, success=False, error=e)
            raise


class FuncCall:
    def __init__(self, model_name=None):
        api_logger.info("=== Initializing FuncCall ===")
        log_environment_info()
        
        self.model_name = model_name
        token_model = 'gpt-4'
        self.encoding = tiktoken.encoding_for_model(token_model)
        
        api_logger.info(f"FuncCall initialized with model: {model_name}")
        api_logger.info(f"Using tokenizer for: {token_model}")

    def message_construction(self, prompt, model_name=""):
        api_logger.info(f"=== Constructing messages for model: {model_name} ===")
        
        if model_name != 'gemini':
            messages = [{"role": "user", "content": prompt}]
            api_logger.info("Using standard OpenAI message format")
        else:
            messages = [{"role": "user", "parts": [prompt]}]
            api_logger.info("Using Gemini message format")
        
        api_logger.info(f"Constructed {len(messages)} messages")
        return messages

    def llm_func(self, messages, model_name):
        api_logger.info("=== FuncCall.llm_func Called ===")
        api_logger.info(f"Target model: {model_name}")
        
        # Convert any numpy arrays in messages to JSON-serializable format
        try:
            original_messages_str = str(messages)
            messages = make_json_serializable(messages)
            new_messages_str = str(messages)
            
            if original_messages_str != new_messages_str:
                api_logger.info("Messages contained numpy arrays - converted to JSON-serializable format")
        except Exception as e:
            api_logger.warning(f"Could not compare messages for numpy detection: {e}")
            messages = make_json_serializable(messages)
        
        max_retries = 5
        
        for attempt in range(1, max_retries + 1):
            api_logger.info(f"=== Attempt {attempt}/{max_retries} ===")
            log_request_details(messages, model_name, attempt)
            
            try:
                result = self._call_legacy_openai(messages, model_name)
                log_response_details(result, success=True)
                return result
                
            except Exception as e:
                log_response_details(None, success=False, error=e)
                print(f"LLM API call failed (attempt {attempt}/{max_retries}): {e}")
                
                if attempt >= max_retries:
                    print(f"Failed to get response after {max_retries} attempts")
                    api_logger.error(f"All {max_retries} attempts failed!")
                    return "Error: Failed to get LLM response"
                
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                api_logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)

    def _call_openai_gpt(self, messages, model_name):
        """调用 OpenAI GPT 模型（使用旧版 API）"""
        return self._call_legacy_openai(messages, model_name)

    def _call_legacy_openai(self, messages, model_name):
        """调用旧版 OpenAI API（兼容所有通过代理访问的模型）"""
        api_logger.info("=== Calling Legacy OpenAI API ===")
        
        # 规范化messages格式，确保符合OpenAI API要求
        # api_logger.info("Normalizing message format for OpenAI API...")
        # messages = normalize_openai_messages(messages)
        
        # 最终检查：确保没有超大内容
        # total_size = sum(len(str(msg.get('content', ''))) for msg in messages)
        # if total_size > 100000:  # 如果总内容超过100k字符
        #     api_logger.error(f"Content too large ({total_size} chars), truncating messages...")
        #     # 截断最长的消息
        #     for msg in messages:
        #         if 'content' in msg and len(str(msg['content'])) > 5000:
        #             original_len = len(str(msg['content']))
        #             msg['content'] = str(msg['content'])[:5000] + f"... (truncated from {original_len} chars)"
        #             api_logger.info(f"Truncated message content from {original_len} to 5000 chars")
        
        # 记录API配置
        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        api_type = os.environ.get("OPENAI_API_TYPE")
        api_version = os.environ.get("OPENAI_API_VERSION", "2023-05-15")
        
        api_logger.info(f"API Base: {api_base}")
        api_logger.info(f"API Key: {api_key[:10] if api_key else 'NOT SET'}...")
        # api_logger.info(f"API Type: {api_type}")
        # api_logger.info(f"API Version: {api_version}")
        
        # 设置OpenAI配置
        openai.api_key = api_key
        openai.api_base = api_base
        
        request_params = {
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 1024
        }
        
        # 检查是否是 Azure API
        if api_type == "azure":
            api_logger.info("Using Azure OpenAI API")
            openai.api_type = "azure"
            openai.api_version = api_version
            request_params["engine"] = model_name
            api_logger.info(f"Azure Engine: {model_name}")
        else:
            api_logger.info("Using standard OpenAI API or compatible")
            request_params["model"] = model_name
            api_logger.info(f"Model: {model_name}")
        
        # 最终验证messages格式
        for i, msg in enumerate(messages):
            if not isinstance(msg.get('content'), str):
                api_logger.error(f"CRITICAL: Message {i} content is still not string: {type(msg.get('content'))}")
        
        # 记录完整的请求参数大小
        try:
            request_str = json.dumps(request_params, ensure_ascii=False, indent=2)
            api_logger.info(f"Final request size: {len(request_str)} characters")
            if len(request_str) > 50000:
                api_logger.warning(f"WARNING: Final request size ({len(request_str)} chars) is still very large!")
        except Exception as e:
            api_logger.error(f"Could not calculate request size: {e}")
        
        try:
            api_logger.info("Sending request to OpenAI API...")
            start_time = time.time()
            
            result = openai.ChatCompletion.create(**request_params)
            
            end_time = time.time()
            api_logger.info(f"Request completed in {end_time - start_time:.2f} seconds")
            
            # 记录完整响应
            api_logger.debug(f"Full API Response: {json.dumps(result, ensure_ascii=False, indent=2)}")
            
            response_content = result["choices"][0]["message"]["content"]
            api_logger.info("API call successful!")
            
            return response_content
            
        except Exception as e:
            api_logger.error(f"OpenAI API call failed: {e}")
            api_logger.error(f"Exception type: {type(e).__name__}")
            
            # 记录详细的错误信息
            if hasattr(e, 'response'):
                api_logger.error(f"Response status: {getattr(e.response, 'status_code', 'Unknown')}")
                api_logger.error(f"Response headers: {getattr(e.response, 'headers', 'Unknown')}")
                api_logger.error(f"Response text: {getattr(e.response, 'text', 'Unknown')}")
            
            raise e

    def _call_openai_compatible_api(self, messages, model_name):
        """调用 OpenAI 兼容的 API（使用旧版 API 格式）"""
        return self._call_legacy_openai(messages, model_name)


def findActionInWebArena_old(agent_response):
    """
    从 WebArena 代理响应中提取动作
    """
    api_logger.debug(f"Extracting action from WebArena response: {agent_response[:200]}...")
    
    # 尝试多种方法提取动作
    
    # 方法1: 查找 "Action:" 关键词
    if "Action:" in agent_response:
        action = agent_response.split("Action:")[-1].strip()
        api_logger.info(f"Found action using 'Action:' method: {action}")
        return action
    
    # 方法2: 查找 ``` 包围的内容
    if "```" in agent_response:
        parts = agent_response.split("```")
        if len(parts) >= 2:
            action = parts[1].strip()
            api_logger.info(f"Found action using '```' method: {action}")
            return action
    
    # 方法3: 查找 "In summary, the next action I will perform is" 后的内容
    summary_phrase = "In summary, the next action I will perform is"
    if summary_phrase in agent_response:
        after_summary = agent_response.split(summary_phrase)[-1].strip()
        if "```" in after_summary:
            action = after_summary.split("```")[1].strip()
            api_logger.info(f"Found action using summary method: {action}")
            return action
    
    # 方法4: 查找常见的 WebArena 动作模式
    action_patterns = [
        r'(click \[\d+\])',
        r'(type \[\d+\] \[.*?\](?:\s*\[0|1\])?)',
        r'(hover \[\d+\])',
        r'(scroll \[(?:up|down)\])',
        r'(goto \[.*?\])',
        r'(press \[.*?\])',
        r'(stop \[.*?\])',
        r'(new_tab)',
        r'(close_tab)',
        r'(go_back)',
        r'(go_forward)',
        r'(tab_focus \[\d+\])'
    ]
    
    for pattern in action_patterns:
        match = re.search(pattern, agent_response, re.IGNORECASE)
        if match:
            api_logger.info(f"Found action using pattern method: {match.group(1)}")
            return match.group(1)
    
    # 方法5: 取最后一行作为动作
    lines = agent_response.strip().split('\n')
    if lines:
        last_line = lines[-1].strip()
        if last_line and not last_line.endswith('.') and not last_line.endswith('?'):
            api_logger.info(f"Found action using last line method: {last_line}")
            return last_line
    
    # 如果都找不到，返回整个响应
    api_logger.warning("Could not extract action using standard methods, returning full response")
    return agent_response.strip()


def perform_test(calling, env, conv, model_name, idx, max_steps):
    api_logger.info(f"=== Starting perform_test for task {idx} ===")
    api_logger.info(f"Model: {model_name}, Max steps: {max_steps}")
    
    Task = os.environ["TASK"]
    model_type = os.environ.get("MODEL_TYPE", "standard")
    dir_path = f"test_result/{Task}/{model_name}_{model_type}"
    file_path = f"{dir_path}/search_results_{idx}.json"

    os.makedirs(dir_path, exist_ok=True)
    done = False
    current_step = 0
    new_env_score = 0
    current_recent_actions = []
    
    print(f"Starting test for task {idx} with model {model_name}")
    api_logger.info(f"Results will be saved to: {file_path}")
    
    while not done and current_step < max_steps:
        current_step += 1
        print(f"Step {current_step}/{max_steps}")
        api_logger.info(f"=== Step {current_step}/{max_steps} ===")
        
        try:
            # 截断对话以适应token限制
            max_len = int(os.environ.get("MAX_TOKEN_LENGTH", 7000))
            api_logger.info(f"Token limit: {max_len}")
            
            # 首先截断消息中的长内容
            for i, msg in enumerate(conv.messages):
                if isinstance(msg, dict) and 'content' in msg:
                    content = str(msg['content'])
                    if len(content) > 10000:  # 如果单条消息超过10k字符
                        # 截断为前5000字符
                        truncated_content = content[:5000] + f"\n... (truncated from {len(content)} chars)"
                        conv.messages[i]['content'] = truncated_content
                        api_logger.info(f"Truncated message {i} from {len(content)} to {len(truncated_content)} characters")
            
            conv_length = len(calling.encoding.encode(str(conv)))
            api_logger.info(f"Current conversation length after content truncation: {conv_length} tokens")
            
            # 然后截断对话历史
            while conv_length > max_len - 60:
                if len(conv.messages) > 4:
                    api_logger.info("Truncating conversation to fit token limit")
                    del conv.messages[4:6]
                    conv_length = len(calling.encoding.encode(str(conv)))
                    api_logger.info(f"New conversation length: {conv_length} tokens")
                else:
                    break
            
            # 获取LLM响应
            prompt = conv.to_openai_api_messages()
            api_logger.info("Getting LLM response...")
            
            # remove the image in every message if exists
            for i in range(len(prompt)):
                if prompt and prompt[i]['role'] == 'user' and isinstance(prompt[i]['content'], dict) and 'text' in prompt[i]['content']:
                    prompt[i]['content'] = prompt[i]['content']['text']
                
            agent_response = calling.llm_func(prompt, model_name)
            print(f"Agent response: {agent_response[:200]}...")
            api_logger.info(f"Agent response received: {agent_response[:200]}...")
            
            # 根据任务类型提取动作
            if Task == "webarena":
                # new_action = findActionInWebArena(agent_response)
                force_prefix = env.agent.prompt_constructor.instruction[
                    "meta_data"
                ].get("force_prefix", "")
                response = f"{force_prefix}{agent_response}"
                parsed_response = env.agent.prompt_constructor.extract_action(
                    response
                )
                new_action = parsed_response
                
            elif Task == "sciworld":
                action_candidate = agent_response.split('Action:')[-1].strip()
                new_action = findValidActionNew([action_candidate], env, env.get_look_around(), current_recent_actions)
            else:
                new_action = agent_response.split('Action:')[-1].strip()
            
            print(f"Extracted action: {new_action}")
            api_logger.info(f"Extracted action: {new_action}")

            # 执行动作
            api_logger.info("Executing action in environment...")
            step_output = env.step(new_action)
            env_state, env_reward, env_done = (
                step_output.state,
                step_output.reward,
                step_output.done,
            )
            
            # 获取当前观察并转换numpy数组
            current_obs = env.observe()
            current_obs = make_json_serializable(current_obs)  # 转换numpy数组为列表
            done = env_done
            new_env_score = env_reward
            
            print(f"Action: {new_action}")
            print(f"State: {str(env_state)[:200]}...")
            print(f"Reward: {env_reward}")
            print(f"Done: {env_done}")
            
            api_logger.info(f"Environment response - State: {str(env_state)[:100]}...")
            api_logger.info(f"Reward: {env_reward}, Done: {env_done}")
            
            # 更新对话
            conv.append_message(conv.roles[1], None)
            conv.update_last_message(agent_response)
            conv.append_message(conv.roles[0], current_obs)

            # 检查是否需要提早结束
            if new_env_score < 0:
                print("Negative reward received, ending episode")
                api_logger.warning("Negative reward received, ending episode")
                done = True
                new_env_score = 0
                
            current_recent_actions.append(f'({new_action}, {current_obs})')
            
        except Exception as e:
            print(f"Error in step {current_step}: {e}")
            api_logger.error(f"Error in step {current_step}: {e}")
            import traceback
            traceback.print_exc()
            api_logger.error(f"Traceback: {traceback.format_exc()}")
            done = True
            break

    print(f"Test completed after {current_step} steps with score {new_env_score}")
    api_logger.info(f"Test completed after {current_step} steps with score {new_env_score}")

    # 保存结果
    final_result = {
        "task_id": idx,
        "env_score": new_env_score,
        "model_name": model_name,
        "step_num": current_step,
        "state": conv.to_openai_api_messages(),
    }
    
    # 确保所有数据都是JSON可序列化的
    final_result = make_json_serializable(final_result)
    
    save_json(final_result, file_path)
    print(f"Results saved to {file_path}")
    api_logger.info(f"Results saved to {file_path}")


def perform_test_revise(calling, env, conv, model_name, idx, max_steps, content_ls):
    api_logger.info(f"=== Starting perform_test_revise for task {idx} ===")
    api_logger.info(f"Model: {model_name}, Max steps: {max_steps}, Content count: {len(content_ls)}")
    
    Task = os.environ["TASK"]
    model_type = os.environ.get("MODEL_TYPE", "standard")
    dir_path = f"revise_result/{Task}/{model_name}_{model_type}"
    file_path = f"{dir_path}/search_results_{idx}.json"
    os.makedirs(dir_path, exist_ok=True)
    done = False
    current_step = 0
    new_env_score = 0
    current_recent_actions = []
    
    api_logger.info(f"Results will be saved to: {file_path}")
    
    # 重放提供的内容
    api_logger.info("=== Replaying provided content ===")
    for i, content in enumerate(content_ls):
        api_logger.info(f"Replaying content {i+1}/{len(content_ls)}")
        agent_response = content
        if Task == "webarena":
            new_action = findActionInWebArena(agent_response)
        elif Task == "sciworld":
            action_candidate = agent_response.split('Action:')[-1].strip()
            new_action = findValidActionNew([action_candidate], env, env.get_look_around(), current_recent_actions)
        else:
            new_action = agent_response.split('Action:')[-1].strip()
            
        api_logger.info(f"Replaying action: {new_action}")
        step_output = env.step(new_action)
        env_state, env_reward, env_done = (
            step_output.state,
            step_output.reward,
            step_output.done,
        )
        current_obs = env.observe()
        current_obs = make_json_serializable(current_obs)  # 转换numpy数组为列表
        done = env_done
        new_env_score = env_reward
        conv.append_message(conv.roles[1], None)
        conv.update_last_message(agent_response)
        conv.append_message(conv.roles[0], current_obs)
        
        api_logger.info(f"Replay step - Reward: {env_reward}, Done: {env_done}")

    # 继续执行剩余步骤
    api_logger.info("=== Continuing with new steps ===")
    from copy import deepcopy
    history = deepcopy(conv)
    max_len = int(os.environ.get("MAX_TOKEN_LENGTH", 7000))
    
    while not done and current_step < max_steps:
        current_step += 1
        api_logger.info(f"=== New Step {current_step}/{max_steps} ===")
        
        conv_length = len(calling.encoding.encode(str(conv)))
        api_logger.info(f"Current conversation length: {conv_length} tokens")
        
        # 首先截断消息中的长内容
        for i, msg in enumerate(conv.messages):
            if isinstance(msg, dict) and 'content' in msg:
                content = str(msg['content'])
                if len(content) > 10000:  # 如果单条消息超过10k字符
                    # 截断为前5000字符
                    truncated_content = content[:5000] + f"\n... (truncated from {len(content)} chars)"
                    conv.messages[i]['content'] = truncated_content
                    api_logger.info(f"Truncated revise message {i} from {len(content)} to {len(truncated_content)} characters")
        
        conv_length = len(calling.encoding.encode(str(conv)))
        api_logger.info(f"Conversation length after content truncation: {conv_length} tokens")
        
        while conv_length > max_len - 60:
            if len(conv.messages) > 4:
                api_logger.info("Truncating conversation to fit token limit")
                del conv.messages[4:6]
                conv_length = len(calling.encoding.encode(str(conv)))
                api_logger.info(f"New conversation length: {conv_length} tokens")
            else:
                break
                
        prompt = conv.to_openai_api_messages()
        api_logger.info("Getting LLM response for new step...")
        agent_response = calling.llm_func(prompt, model_name)
        
        if Task == "webarena":
            new_action = findActionInWebArena(agent_response)
        elif Task == "sciworld":
            action_candidate = agent_response.split('Action:')[-1].strip()
            new_action = findValidActionNew([action_candidate], env, env.get_look_around(), current_recent_actions)
        else:
            new_action = agent_response.split('Action:')[-1].strip()
            
        api_logger.info(f"New step action: {new_action}")
        step_output = env.step(new_action)
        env_state, env_reward, env_done = (
            step_output.state,
            step_output.reward,
            step_output.done,
        )
        current_obs = env.observe()
        current_obs = make_json_serializable(current_obs)  # 转换numpy数组为列表
        done = env_done
        new_env_score = env_reward
        
        conv.append_message(conv.roles[1], None)
        conv.update_last_message(agent_response)
        conv.append_message(conv.roles[0], current_obs)

        history.append_message(conv.roles[1], None)
        history.update_last_message(agent_response)
        history.append_message(conv.roles[0], current_obs)

        if new_env_score < 0:
            api_logger.warning("Negative reward received, ending episode")
            done = True
            new_env_score = 0
            
        print(f"Action: {new_action}")
        print(f"State: {env_state}")
        print(f"Reward: {env_reward}")
        api_logger.info(f"New step - Reward: {env_reward}, Done: {env_done}")
        current_recent_actions.append(f'({new_action}, {current_obs})')

    api_logger.info(f"Test revise completed after {current_step} new steps with score {new_env_score}")

    final_result = {
        "task_id": idx,
        "env_score": new_env_score,
        "model_name": model_name,
        "step_num": current_step,
        "state": history.to_openai_api_messages(),
    }
    
    # 确保所有数据都是JSON可序列化的
    final_result = make_json_serializable(final_result)
    
    save_json(final_result, file_path)
    api_logger.info(f"Results saved to {file_path}")


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
    "I recognize my mistake. Let's find a better solution.",
    "My judgment was incorrect. I need to rethink it.",
    "I made an error. I must determine how to correct it.",
    "I acted hastily. I need to reconsider my choices.",
    "I misjudged the scenario. Time to reflect and adjust.",
    "My initial steps were wrong. I need a new approach.",
    "I realize I chose poorly. I must change direction.",
    "I overlooked something important. I need to address it.",
    "I miscalculated. It's time to figure out a better way.",
    "I made a poor decision. I need to set things right.",
    "I recognize my failure. I need to learn and move forward.",
    "I didn't think this through. I must reevaluate.",
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

# 启动时记录环境信息
log_environment_info()
api_logger.info("LLM API Logger module loaded successfully")











# """ 原版
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
# """
# import os
# import json
# import csv
# import os
# import openai
# import tiktoken
# from transformers import AutoTokenizer
# from mcts_utils.sciworld.eval_utils_sw import findValidActionNew


# class FuncCallOffline:
#     def __init__(self, model_name=None):
#         from vllm import LLM, SamplingParams
#         self.model_name = model_name
#         self.llm = LLM(model=os.environ["MODEL_DIR"], dtype="half")
#         if "TEMP" in os.environ:
#             print(f'当前 TEMP 值: {os.environ["TEMP"]}')
#         else:
#             os.environ["TEMP"] = "1"
#             print("TEMP 不存在，已设置为 1")
#         self.sampling_params = SamplingParams(temperature=float(os.environ["TEMP"]), max_tokens=500, stop=["<|eot_id|>"])
#         tokenizer = AutoTokenizer.from_pretrained(os.environ["MODEL_DIR"])
#         self.encoding = tokenizer

#     def llm_func(self, messages, model_name):
#         outputs = self.llm.chat(messages, sampling_params=self.sampling_params)
#         text = outputs[0].outputs[0].text.strip()
#         return text
    

# class FuncCall:
#     def __init__(self, model_name=None):
#         self.model_name = model_name
#         token_model = 'gpt-4'
#         self.encoding = tiktoken.encoding_for_model(token_model)

#     def message_construction(self, prompt, model_name=""):
#         if model_name != 'gemini':
#             messages = [{"role": "user", "content": prompt}]
#         else:
#             messages = [{"role": "user", "parts": [prompt]}]
#         return messages

#     def llm_func(self, messages, model_name):
#         ind = 0
#         while True:
#             try:
#                 if "gpt" in model_name:
#                     openai.api_key = os.environ["OPENAI_API_KEY"]
#                     openai.api_base = os.environ["OPENAI_API_BASE"]
#                     openai.api_type = "azure"
#                     openai.api_version = os.environ["OPENAI_API_VERSION"]
#                     result = openai.ChatCompletion.create(
#                         engine=model_name,
#                         messages=messages,
#                         temperature=0,
#                         stop=None)
#                     clean_result = result["choices"][0]["message"]["content"]
#                 return clean_result
#             except Exception as e:
#                 if ind > 100000:
#                     return -1
#                 ind += 1
#                 continue


# def perform_test(calling, env, conv, model_name, idx, max_steps):
#     Task = os.environ["TASK"]
#     model_type = os.environ["MODEL_TYPE"]
#     dir_path = f"test_result/{Task}/{model_name}_{model_type}"
#     file_path = f"{dir_path}/search_results_{idx}.json"

#     os.makedirs(dir_path, exist_ok=True)
#     done = False
#     max_steps = max_steps
#     current_step = 0
#     new_env_score = 0
#     current_recent_actions = []
#     while not done:
#         if current_step >= max_steps:
#             done = True
#             continue

#         max_len = 7000
#         if "MAX_TOKEN_LENGTH" in os.environ:
#             max_len = int(os.environ["MAX_TOKEN_LENGTH"])
#         while len(calling.encoding.encode(str(conv))) > max_len - 60:
#             del conv.messages[4:6]
#         current_step += 1
#         prompt = conv.to_openai_api_messages()
#         agent_response = calling.llm_func(prompt, model_name)
#         # exit()
#         # if Task == "webarena":
#         #     new_action = findActionInWebArena(agent_response) # TODO: Implement findActionInWebArena
#         new_action = agent_response.split('Action:')[-1].strip()
#         if Task == "sciworld":
#             new_action = findValidActionNew([new_action], env, env.get_look_around(), current_recent_actions)

#         step_output = env.step(new_action)
#         env_state, env_reward, env_done = (
#                         step_output.state,
#                         step_output.reward,
#                         step_output.done,
#                     )
#         current_obs = env.observe()
#         done = env_done
#         new_env_score = env_reward
#         conv.append_message(conv.roles[1], None)

#         conv.update_last_message(agent_response)
#         conv.append_message(conv.roles[0], current_obs)

#         if new_env_score < 0:
#             done = True
#             new_env_score = 0
#         print(new_action)
#         print(env_state)
#         print(env_reward)
#         current_recent_actions.append(f'({new_action}, {current_obs})')


#     final_result = {
#         "task_id": idx,
#         "env_score": new_env_score,
#         "model_name": model_name,
#         "step_num": current_step,
#         "state": conv.to_openai_api_messages(),
#         }
    
#     save_json(final_result, file_path)

# def perform_test_revise(calling, env, conv, model_name, idx, max_steps, content_ls):
#     Task = os.environ["TASK"]
#     model_type = os.environ["MODEL_TYPE"]
#     dir_path = f"revise_result/{Task}/{model_name}_{model_type}"
#     file_path = f"{dir_path}/search_results_{idx}.json"
#     os.makedirs(dir_path, exist_ok=True)
#     done = False
#     max_steps = max_steps
#     current_step = 0
#     new_env_score = 0
#     current_recent_actions = []
#     for content in content_ls:
#         agent_response = content
#         new_action = agent_response.split('Action:')[-1].strip()
#         if Task == "sciworld":
#             new_action = findValidActionNew([new_action], env, env.get_look_around(), current_recent_actions)
#         step_output = env.step(new_action)
#         env_state, env_reward, env_done = (
#                         step_output.state,
#                         step_output.reward,
#                         step_output.done,
#                     )
#         current_obs = env.observe()
#         done = env_done
#         new_env_score = env_reward
#         conv.append_message(conv.roles[1], None)
#         conv.update_last_message(agent_response)
#         conv.append_message(conv.roles[0], current_obs)

#     from copy import deepcopy
#     history = deepcopy(conv)
#     max_len = 7000
#     if "MAX_TOKEN_LENGTH" in os.environ:
#         max_len = int(os.environ["MAX_TOKEN_LENGTH"])
#     while not done:
#         if current_step >= max_steps:
#             done = True
#             continue
#         while len(calling.encoding.encode(str(conv))) > max_len - 60:
#             del conv.messages[4:6]
#         current_step += 1
#         prompt = conv.to_openai_api_messages()
#         agent_response = calling.llm_func(prompt, model_name)
#         new_action = agent_response.split('Action:')[-1].strip()
#         if Task == "sciworld":
#             new_action = findValidActionNew([new_action], env, env.get_look_around(), current_recent_actions)
#         step_output = env.step(new_action)
#         env_state, env_reward, env_done = (
#                         step_output.state,
#                         step_output.reward,
#                         step_output.done,
#                     )
#         current_obs = env.observe()
#         done = env_done
#         new_env_score = env_reward
#         conv.append_message(conv.roles[1], None)
#         conv.update_last_message(agent_response)
#         conv.append_message(conv.roles[0], current_obs)

#         history.append_message(conv.roles[1], None)
#         history.update_last_message(agent_response)
#         history.append_message(conv.roles[0], current_obs)

#         if new_env_score < 0:
#             done = True
#             new_env_score = 0
#         print(new_action)
#         print(env_state)
#         print(env_reward)
#         current_recent_actions.append(f'({new_action}, {current_obs})')


#     final_result = {
#         "task_id": idx,
#         "env_score": new_env_score,
#         "model_name": model_name,
#         "step_num": current_step,
#         "state": history.to_openai_api_messages(),
#         }
    
#     save_json(final_result, file_path)

# def get_last_processed_index(progress_file):
#     """Retrieve the last processed index from the progress file."""
#     if os.path.exists(progress_file):
#         with open(progress_file, 'r', encoding='utf-8') as f:
#             last_index = f.read().strip()
#             return int(last_index) if last_index else 0
#     else:
#         return 0


# def update_progress(progress_file, index):
#     """Update the last processed index in the progress file."""
#     with open(progress_file, 'w', encoding='utf-8') as f:
#         f.write(str(index))


# def read_jsonline(address):
#     not_mark = []
#     with open(address, 'r', encoding="utf-8") as f:
#         for jsonstr in f.readlines():
#             jsonstr = json.loads(jsonstr)
#             not_mark.append(jsonstr)
#     return not_mark


# def read_json(address):
#     with open(address, 'r', encoding='utf-8') as json_file:
#         json_data = json.load(json_file)
#     return json_data

# import os

# def create_file_if_not_exists(file_path):
#     if not os.path.exists(file_path):
#         with open(file_path, 'w') as file:
#             pass  # 创建空文件

# def read_jsonl(address):
#     not_mark = []
#     with open(address, "r", encoding="utf8") as f:
#         for item in jsonlines.Reader(f):
#             not_mark.append(item)
#     return not_mark


# def read_csv(address):
#     dataset = []
#     with open(address, encoding='utf-8-sig') as f:
#         for row in csv.reader(f, skipinitialspace=True):
#             dataset.append(row)
#     return dataset


# def read_tsv(address):
#     dataset = []
#     with open(address, encoding='utf-8-sig') as f:
#         tsvreader = csv.reader(f, delimiter='\t')
#         for row in tsvreader:
#             dataset.append(row)
#     return dataset


# def read_txt(address, sep):
#     dataset = []
#     with open(address, 'r', encoding="utf-8") as f:
#         for data in f.readlines():
#             data = data.replace('\n', '').split(sep)
#             dataset.append(data)
#     return dataset


# def save_jsonline(ls, address):
#     for item in ls:
#         with open(address, 'a+', encoding='utf-8') as f:
#             line = json.dumps(item, ensure_ascii=False)
#             f.write(line + '\n')


# def save_json(ls, address):
#     #json_str = json.dumps(ls, indent=4)
#     with open(address, 'w', encoding='utf-8') as json_file:
#         json.dump(ls, json_file, ensure_ascii=False, indent=4)


# def sort_dic(dic):
#     dic = sorted(dic.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
#     return dic


# def expand_dataset(dataset, expand_time):
#     final_dataset = []
#     for item in dataset:
#         for i in range(expand_time):
#             final_dataset.append(item)
#     return final_dataset

# def rewrite(dataset, output_path):
#     final_dataset = []
#     for data in dataset:
#         conversation = []
#         revise_log = data["revise_log"]
#         temp = {
#             "system": revise_log[0]["content"],
#             "input": revise_log[1]["content"],
#             "output": revise_log[2]["content"],
#         }
#         conversation.append(temp)
#         i = 3
#         while i+1 < len(revise_log):
#             temp = {
#                 "input": revise_log[i]["content"],
#                 "output": revise_log[i+1]["content"],
#             }
#             i += 2
#             conversation.append(temp)
#         final_dataset.append({"conversation": conversation})

#     save_json(final_dataset, output_path)

# def write_to_jsonl(output_file, data):
#     """
#     Appends data to a JSONL file.

#     Args:
#         output_file: Path to the JSONL file.
#         data: Data to append as a dictionary.
#     """
#     with open(output_file, 'a+', encoding='utf-8') as jsonl_file:
#         jsonl_file.write(json.dumps(data, ensure_ascii=False) + '\n')

# revision_thoughts = [
#     "I realize my approach was flawed. I need to revise it.",
#     "I took the wrong steps. I need to identify the right path.",
#     "My actions were incorrect. I must adjust my strategy.",
#     "I see an error in my actions. I need to fix it.",
#     "I misunderstood the situation. Time to reassess.",
#     "My decision was wrong. I should reevaluate.",
#     "I went off course. I need to steer back on track.",
#     "I recognize my mistake. Let’s find a better solution.",
#     "My judgment was incorrect. I need to rethink it.",
#     "I made an error. I must determine how to correct it.",
#     "I acted hastily. I need to reconsider my choices.",
#     "I misjudged the scenario. Time to reflect and adjust.",
#     "My initial steps were wrong. I need a new approach.",
#     "I realize I chose poorly. I must change direction.",
#     "I overlooked something important. I need to address it.",
#     "I miscalculated. It’s time to figure out a better way.",
#     "I made a poor decision. I need to set things right.",
#     "I recognize my failure. I need to learn and move forward.",
#     "I didn’t think this through. I must reevaluate.",
#     "I strayed from the goal. I need to realign my efforts."
# ]

# prompt_template = """
# You are a good verifier. You will be given a log that records an agent interacting with an environment to solve a science task. The format of the log is:
# ```
# Action: #Action
# Observation: #Observation
# ```

# You need to verify whether the current action is good, bad, or uncertain in the log. 
# - A **good** action is greatly helpful to solve the task.
# - A **bad** action is greatly harmful to solve the task.
# - An **uncertain** action is one that is neither good nor bad, or you cannot judge based on the current information.

# Log:
# Task Description: {task_description}
# {action_obs_prompt}
# Current_Action: {action}
# Current_Observation: {observation}

# You must give reasons first and then provide the response in the format: Judgement: <Good or Bad or Uncertain>
# """.strip()

