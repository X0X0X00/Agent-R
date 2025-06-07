"""
Script to run end-to-end evaluation on the benchmark

代码概述
这是一个用于测试AI代理在网页环境中完成任务能力的评估框架。
代码让AI模型 (如Gemini 2.5 Flash 或 GPT-4o-mini)控制一个浏览器，尝试完成特定任务，然后评估其表现。

脚本的核心逻辑结构
1. 配置解析 (config)
2. 环境与 agent 构建 (construct_agent + ScriptBrowserEnv)
3. 多任务迭代测试 (test)
4. 每个任务中循环交互 (trajectory生成)
5. 执行评估器 (evaluator)
6. 保存日志/结果/trace

"""

"""
Script to run end-to-end evaluation on the benchmark

代码概述
这是一个用于测试AI代理在网页环境中完成任务能力的评估框架。
代码让AI模型 (如Gemini 2.5 Flash 或 GPT-4o-mini)控制一个浏览器，尝试完成特定任务，然后评估其表现。

脚本的核心逻辑结构
1. 配置解析 (config)
2. 环境与 agent 构建 (construct_agent + ScriptBrowserEnv)
3. 多任务迭代测试 (test)
4. 每个任务中循环交互 (trajectory生成)
5. 执行评估器 (evaluator)
6. 保存日志/结果/trace

"""

import os


# API 提供商

# API 提供商
# openAI gpt-4o-mini
# os.environ["OPENAI_API_KEY"] = "sk-iNUPz3eXLdha41JD14E3CaAaF4Fb4f11927723Dc8eBa2eA3"
# os.environ["OPENAI_API_BASE"] = "http://10.112.59.240:3001/v1"

# Google google/gemini-2.5-flash
# Google google/gemini-2.5-flash
os.environ["OPENAI_API_KEY"] = "sk-yE4vSQpw9eluzB4DBdB03eBa226a4f668b1aBcC3FeAb10A5"
os.environ["OPENAI_API_BASE"] = "http://10.112.59.240:3001/v1"

# export OPENAI_API_BASE=http://10.112.59.240:3001/v1
# export OPENAI_API_KEY=sk-yE4vSQpw9eluzB4DBdB03eBa226a4f668b1aBcC3FeAb10A5

# 贵的服务商
# os.environ["OPENAI_API_KEY"] = "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"
# os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# export OPENAI_API_BASE=http://10.112.59.240:3001/v1
# export OPENAI_API_KEY=sk-yE4vSQpw9eluzB4DBdB03eBa226a4f668b1aBcC3FeAb10A5

# 贵的服务商
# os.environ["OPENAI_API_KEY"] = "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"
# os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"


import argparse
import glob
import json
import logging
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path
import openai

# 导入 WebArena 组件和 agent 构建逻辑
from agent import (
    Agent,
    PromptAgent,
    TeacherForcingAgent,
    construct_agent,
)
from agent.prompts import *
from browser_env import (
    Action,
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)
from browser_env.actions import is_equivalent
from browser_env.auto_login import get_site_comb_from_filepath
from browser_env.helper_functions import (
    RenderHelper,
    get_action_description,
)
from evaluation_harness import evaluator_router



# 日志初始化逻辑
LOG_FOLDER = "log_files" # 日志文件夹


# 日志初始化逻辑
LOG_FOLDER = "log_files" # 日志文件夹
Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
LOG_FILE_NAME = f"{LOG_FOLDER}/log_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_{random.randint(0, 10000)}.log"

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOG_FILE_NAME)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# Set the log format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# 参数解析器
# 参数解析器
def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the browser"
    )
    parser.add_argument(
        "--slow_mo",
        type=int,
        default=0,
        help="Slow down the browser by the specified amount",
    )
    parser.add_argument(
        "--action_set_tag", default="id_accessibility_tree", help="Action type"
    )
    parser.add_argument(
        "--observation_type",
        choices=["accessibility_tree", "html", "image"],
        default="accessibility_tree",
        help="Observation type",
    )
    parser.add_argument(
        "--current_viewport_only",
        action="store_true",
        help="Only use the current viewport for the observation",
    )
    parser.add_argument("--viewport_width", type=int, default=1280)
    parser.add_argument("--viewport_height", type=int, default=720)
    parser.add_argument("--save_trace_enabled", action="store_true")
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)

    # parser.add_argument("--max_steps", type=int, default=30) 可以把 max_steps: 从30增至50（给予更多探索机会）
    # parser.add_argument("--max_steps", type=int, default=30) 可以把 max_steps: 从30增至50（给予更多探索机会）
    parser.add_argument("--max_steps", type=int, default=30)

    # agent config
    parser.add_argument("--agent_type", type=str, default="prompt")
    parser.add_argument(
        "--instruction_path",
        type=str,
        # default="agents/prompts/state_action_agent.json" 原始版本使用相对路径，现在使用完整的绝对路径。
        # default="agents/prompts/state_action_agent.json" 原始版本使用相对路径，现在使用完整的绝对路径。
        default="/home/ubuntu/zhangzhenhao/Agent-R/AgentGym/agentenv-webarena/webarena/agent/prompts/jsons/p_cot_id_actree_2s.json",
    )
    parser.add_argument(
        "--parsing_failure_th",
        help="When concesecutive parsing failure exceeds this threshold, the agent will stop",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--repeating_action_failure_th",
        help="When concesecutive repeating action exceeds this threshold, the agent will stop",
        type=int,
        default=3,
    )

    # lm config
    parser.add_argument("--provider", type=str, default="openai")
    # parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0613") 原始版本默认模型为 "gpt-3.5-turbo-0613"，修改版本使用 "gpt-4o-mini" / "gemini-2.5-flash-preview"
   
    # parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--model", type=str, default="google/gemini-2.5-flash-preview")
   
    # parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0613") 原始版本默认模型为 "gpt-3.5-turbo-0613"，修改版本使用 "gpt-4o-mini" / "gemini-2.5-flash-preview"
   
    # parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--model", type=str, default="google/gemini-2.5-flash-preview")
   
    parser.add_argument("--mode", type=str, default="chat")
    # parser.add_argument("--temperature", type=float, default=1.0) temperature 原代码为1 
    parser.add_argument("--temperature", type=float, default=0.2) # 降低随机性
    # parser.add_argument("--temperature", type=float, default=1.0) temperature 原代码为1 
    parser.add_argument("--temperature", type=float, default=0.2) # 降低随机性
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--context_length", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=384)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument(
        "--max_retry",
        type=int,
        help="max retry times to perform generations when parsing fails",
        default=1,
    )
    parser.add_argument(
        "--max_obs_length",
        type=int,
        help="when not zero, will truncate the observation to this length before feeding to the model",
        default=1920,
    )
    parser.add_argument(
        "--model_endpoint",
        help="huggingface model endpoint",
        type=str,
        default="",
    )

    # example config
    
    
    parser.add_argument("--test_start_idx", type=int, default=0)
    # parser.add_argument("--test_end_idx", type=int, default=1000) 原始版本默认测试范围为 0-1000，修改版本缩小为 0-50。
    parser.add_argument("--test_end_idx", type=int, default=20)
    # parser.add_argument("--test_end_idx", type=int, default=1000) 原始版本默认测试范围为 0-1000，修改版本缩小为 0-50。
    parser.add_argument("--test_end_idx", type=int, default=20)

    # logging related
    # parser.add_argument("--result_dir", type=str, default="") 原始版本默认结果目录为空字符串，修改版本设置为 "results/gpt-4o-mini"

    # parser.add_argument("--result_dir", type=str, default="results/gpt-4o-mini")
    parser.add_argument("--result_dir", type=str, default="results/gemini-2.5-flash-preview")

    # parser.add_argument("--result_dir", type=str, default="") 原始版本默认结果目录为空字符串，修改版本设置为 "results/gpt-4o-mini"

    # parser.add_argument("--result_dir", type=str, default="results/gpt-4o-mini")
    parser.add_argument("--result_dir", type=str, default="results/gemini-2.5-flash-preview")

    args = parser.parse_args()
    # check the whether the action space is compatible with the observation space
    if (
        args.action_set_tag == "id_accessibility_tree"
        and args.observation_type != "accessibility_tree"
    ):
        raise ValueError(
            f"Action type {args.action_set_tag} is incompatible with the observation type {args.observation_type}"
        )

    return args

# 定义早停函数，检查是否需要提前停止代理的执行。
# 早停条件：达到最大步数、连续解析失败、连续重复动作。
# 定义早停函数，检查是否需要提前停止代理的执行。
# 早停条件：达到最大步数、连续解析失败、连续重复动作。
def early_stop(
    trajectory: Trajectory, max_steps: int, thresholds: dict[str, int]
) -> tuple[bool, str]:
    """Check whether need to early stop"""

    # reach the max step
    num_steps = (len(trajectory) - 1) / 2
    if num_steps >= max_steps:
        return True, f"Reach max steps {max_steps}"

    last_k_actions: list[Action]
    action_seq: list[Action]

    # Case: parsing failure for k times
    k = thresholds["parsing_failure"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    if len(last_k_actions) >= k:
        if all(
            [
                action["action_type"] == ActionTypes.NONE
                for action in last_k_actions
            ]
        ):
            return True, f"Failed to parse actions for {k} times"

    # Case: same action for k times
    k = thresholds["repeating_action"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    action_seq = trajectory[1::2]  # type: ignore[assignment]

    if len(action_seq) == 0:
        return False, ""

    last_action: Action = action_seq[-1]

    if last_action["action_type"] != ActionTypes.TYPE:
        if len(last_k_actions) >= k:
            if all(
                [
                    is_equivalent(action, last_action)
                    for action in last_k_actions
                ]
            ):
                return True, f"Same action for {k} times"

    else:
        # check the action sequence
        if (
            sum([is_equivalent(action, last_action) for action in action_seq])
            >= k
        ):
            return True, f"Same typing action for {k} times"

    return False, ""

# # 原版有bug
# def test(
#     args: argparse.Namespace,
#     agent: Agent | PromptAgent | TeacherForcingAgent,
#     config_file_list: list[str],
# ) -> None:
#     scores = []
#     max_steps = args.max_steps

#     early_stop_thresholds = {
#         "parsing_failure": args.parsing_failure_th,
#         "repeating_action": args.repeating_action_failure_th,
#     }

#     # 环境构建
#     env = ScriptBrowserEnv(
#         headless=not args.render,
#         slow_mo=args.slow_mo,
#         observation_type=args.observation_type,
#         current_viewport_only=args.current_viewport_only,
#         viewport_size={
#             "width": args.viewport_width,
#             "height": args.viewport_height,
#         },
#         save_trace_enabled=args.save_trace_enabled,
#         sleep_after_execution=args.sleep_after_execution,
#     )
#     all_logs = dict()
#     base_path = Path("/home/ubuntu/zhangzhenhao/Agent-R/AgentGym/agentenv-webarena/webarena/")
#     log_file = base_path / f'results/log-{args.model.replace("/", "-")}-{time.strftime("%Y-%m-%d-%H-%M-%S")}.json'
    
#     # 遍历任务配置文件列表
#     for config_file in config_file_list:
#         try:
#             config_file = str(base_path / config_file)
#             render_helper = RenderHelper(
#                 config_file, args.result_dir, args.action_set_tag
#             )

#             all_logs[str(config_file)] = list()

#             # get intent
#             with open(config_file) as f:
#                 _c = json.load(f)
#                 intent = _c["intent"]
#                 task_id = _c["task_id"]
#                 # automatically login
#                 if _c["storage_state"]:
#                     cookie_file_name = os.path.basename(_c["storage_state"])
#                     comb = get_site_comb_from_filepath(cookie_file_name)
#                     temp_dir = tempfile.mkdtemp()
#                     # subprocess to renew the cookie
#                     subprocess.run(
#                         [
#                             "python",
#                             "-m", "browser_env.auto_login",
#                             "--auth_folder",
#                             temp_dir,
#                             "--site_list",
#                             *comb,
#                         ]
#                     )
#                     _c["storage_state"] = f"{temp_dir}/{cookie_file_name}"
#                     assert os.path.exists(_c["storage_state"])
#                     # update the config file
#                     config_file = f"{temp_dir}/{os.path.basename(config_file)}"
#                     with open(config_file, "w") as f:
#                         json.dump(_c, f)

#             # all_logs[str(config_file)] = list()
#             # # get intent
#             # with open(config_file) as f:
#             #     _c = json.load(f)
#             #     intent = _c["intent"]
#             #     task_id = _c["task_id"]

#             #     # 替换
#             #     if _c["storage_state"]:
#             #         cookie_path = str(base_path / _c["storage_state"])
#             #         assert os.path.exists(cookie_path), f"Missing cookie: {_c['storage_state']}"

#                 # 临时修改 
#                 # automatically login

#                 # if _c["storage_state"]:
#                 #     cookie_file_name = os.path.basename(_c["storage_state"])
#                 #     comb = get_site_comb_from_filepath(cookie_file_name)
#                 #     temp_dir = tempfile.mkdtemp()
#                 #     # subprocess to renew the cookie
#                 #     subprocess.run(
#                 #         [
#                 #             "python",
#                 #             "browser_env/auto_login.py",
#                 #             "--auth_folder",
#                 #             temp_dir,
#                 #             "--site_list",
#                 #             *comb,
#                 #         ]
#                 #     )
#                 #     _c["storage_state"] = f"{temp_dir}/{cookie_file_name}"
#                 #     assert os.path.exists(_c["storage_state"])
#                 #     # update the config file
#                 #     config_file = f"{temp_dir}/{os.path.basename(config_file)}"
#                 #     with open(config_file, "w") as f:
#                 #         json.dump(_c, f)

#             logger.info(f"[Config file]: {config_file}")
#             logger.info(f"[Intent]: {intent}")


#             # agent 重置并初始化
#             agent.reset(config_file)

#             # 交互轨迹生成
#             trajectory: Trajectory = []
#             obs, info = env.reset(options={"config_file": config_file})
#             state_info: StateInfo = {"observation": obs, "info": info}
#             trajectory.append(state_info)

#             meta_data = {"action_history": ["None"]}
#             while True:
#                 early_stop_flag, stop_info = early_stop(
#                     trajectory, max_steps, early_stop_thresholds
#                 )

#                 # early stop 判定（防止死循环）
#                 if early_stop_flag:
#                     action = create_stop_action(f"Early stop: {stop_info}")
#                 else:
#                     try:
#                         action = agent.next_action(
#                             trajectory, intent, meta_data=meta_data
#                         )
#                     except ValueError as e:
#                         # get the error message
#                         action = create_stop_action(f"ERROR: {str(e)}")

#                 trajectory.append(action)

#                 action_str = get_action_description(
#                     action,
#                     state_info["info"]["observation_metadata"],
#                     action_set_tag=args.action_set_tag,
#                     prompt_constructor=agent.prompt_constructor
#                     if isinstance(agent, PromptAgent)
#                     else None,
#                 )
#                 all_logs[str(config_file)].append(action_str)
                
#                 # 渲染与日志记录
#                 render_helper.render(
#                     action, state_info, meta_data, args.render_screenshot
#                 )
#                 meta_data["action_history"].append(action_str)

#                 if action["action_type"] == ActionTypes.STOP:
#                     break

#                 obs, _, terminated, _, info = env.step(action)
#                 state_info = {"observation": obs, "info": info}
#                 trajectory.append(state_info)
                
#                 with open(log_file, "w") as f:
#                     json.dump(all_logs, f, indent=4)

#                 if terminated:
#                     # add a action place holder
#                     trajectory.append(create_stop_action(""))
#                     break

#             # 评估器执行
#             evaluator = evaluator_router(config_file)
#             score = evaluator(
#                 trajectory=trajectory,
#                 config_file=config_file,
#                 page=env.page,
#                 client=env.get_page_client(env.page),
#             )

#             scores.append(score)

#             if score == 1:
#                 logger.info(f"[Result] (PASS) {config_file}")
#             else:
#                 logger.info(f"[Result] (FAIL) {config_file}")
#                 logger.info("没有任务成功完成 - scores列表为空")


#             if args.save_trace_enabled:
#                 env.save_trace(
#                     Path(args.result_dir) / "traces" / f"{task_id}.zip"
#                 )

#         except openai.error.OpenAIError as e:
#             # import traceback
#             logger.info(f"[OpenAI Error] {repr(e)}")
#             with open(Path(args.result_dir) / "error.txt", "a") as f:
#                 f.write(f"[OpenAI file]: {config_file}\n")
#                 f.write(f"[OpenAI Error] {repr(e)}\n")
#                 # f.write(traceback.format_exc())  # write stack trace to file
                
#         except Exception as e:
#             logger.info(f"[Unhandled Error] {repr(e)}]")
#             # raise e
#             # import traceback
#             # write to error file
#             with open(Path(args.result_dir) / "error.txt", "a") as f:
#                 f.write(f"[Config file]: {config_file}\n")
#                 f.write(f"[Unhandled Error] {repr(e)}\n")
#                 # f.write(traceback.format_exc())  # write stack trace to file

#         render_helper.close()

#     env.close()
#     logger.info(f"Average score: {sum(scores) / len(scores)}")


# # 原版有bug
# def test(
#     args: argparse.Namespace,
#     agent: Agent | PromptAgent | TeacherForcingAgent,
#     config_file_list: list[str],
# ) -> None:
#     scores = []
#     max_steps = args.max_steps

#     early_stop_thresholds = {
#         "parsing_failure": args.parsing_failure_th,
#         "repeating_action": args.repeating_action_failure_th,
#     }

#     # 环境构建
#     env = ScriptBrowserEnv(
#         headless=not args.render,
#         slow_mo=args.slow_mo,
#         observation_type=args.observation_type,
#         current_viewport_only=args.current_viewport_only,
#         viewport_size={
#             "width": args.viewport_width,
#             "height": args.viewport_height,
#         },
#         save_trace_enabled=args.save_trace_enabled,
#         sleep_after_execution=args.sleep_after_execution,
#     )
#     all_logs = dict()
#     base_path = Path("/home/ubuntu/zhangzhenhao/Agent-R/AgentGym/agentenv-webarena/webarena/")
#     log_file = base_path / f'results/log-{args.model.replace("/", "-")}-{time.strftime("%Y-%m-%d-%H-%M-%S")}.json'
    
#     # 遍历任务配置文件列表
#     for config_file in config_file_list:
#         try:
#             config_file = str(base_path / config_file)
#             render_helper = RenderHelper(
#                 config_file, args.result_dir, args.action_set_tag
#             )

#             all_logs[str(config_file)] = list()

#             # get intent
#             with open(config_file) as f:
#                 _c = json.load(f)
#                 intent = _c["intent"]
#                 task_id = _c["task_id"]
#                 # automatically login
#                 if _c["storage_state"]:
#                     cookie_file_name = os.path.basename(_c["storage_state"])
#                     comb = get_site_comb_from_filepath(cookie_file_name)
#                     temp_dir = tempfile.mkdtemp()
#                     # subprocess to renew the cookie
#                     subprocess.run(
#                         [
#                             "python",
#                             "-m", "browser_env.auto_login",
#                             "--auth_folder",
#                             temp_dir,
#                             "--site_list",
#                             *comb,
#                         ]
#                     )
#                     _c["storage_state"] = f"{temp_dir}/{cookie_file_name}"
#                     assert os.path.exists(_c["storage_state"])
#                     # update the config file
#                     config_file = f"{temp_dir}/{os.path.basename(config_file)}"
#                     with open(config_file, "w") as f:
#                         json.dump(_c, f)

#             # all_logs[str(config_file)] = list()
#             # # get intent
#             # with open(config_file) as f:
#             #     _c = json.load(f)
#             #     intent = _c["intent"]
#             #     task_id = _c["task_id"]

#             #     # 替换
#             #     if _c["storage_state"]:
#             #         cookie_path = str(base_path / _c["storage_state"])
#             #         assert os.path.exists(cookie_path), f"Missing cookie: {_c['storage_state']}"

#                 # 临时修改 
#                 # automatically login

#                 # if _c["storage_state"]:
#                 #     cookie_file_name = os.path.basename(_c["storage_state"])
#                 #     comb = get_site_comb_from_filepath(cookie_file_name)
#                 #     temp_dir = tempfile.mkdtemp()
#                 #     # subprocess to renew the cookie
#                 #     subprocess.run(
#                 #         [
#                 #             "python",
#                 #             "browser_env/auto_login.py",
#                 #             "--auth_folder",
#                 #             temp_dir,
#                 #             "--site_list",
#                 #             *comb,
#                 #         ]
#                 #     )
#                 #     _c["storage_state"] = f"{temp_dir}/{cookie_file_name}"
#                 #     assert os.path.exists(_c["storage_state"])
#                 #     # update the config file
#                 #     config_file = f"{temp_dir}/{os.path.basename(config_file)}"
#                 #     with open(config_file, "w") as f:
#                 #         json.dump(_c, f)

#             logger.info(f"[Config file]: {config_file}")
#             logger.info(f"[Intent]: {intent}")


#             # agent 重置并初始化
#             agent.reset(config_file)

#             # 交互轨迹生成
#             trajectory: Trajectory = []
#             obs, info = env.reset(options={"config_file": config_file})
#             state_info: StateInfo = {"observation": obs, "info": info}
#             trajectory.append(state_info)

#             meta_data = {"action_history": ["None"]}
#             while True:
#                 early_stop_flag, stop_info = early_stop(
#                     trajectory, max_steps, early_stop_thresholds
#                 )

#                 # early stop 判定（防止死循环）
#                 if early_stop_flag:
#                     action = create_stop_action(f"Early stop: {stop_info}")
#                 else:
#                     try:
#                         action = agent.next_action(
#                             trajectory, intent, meta_data=meta_data
#                         )
#                     except ValueError as e:
#                         # get the error message
#                         action = create_stop_action(f"ERROR: {str(e)}")

#                 trajectory.append(action)

#                 action_str = get_action_description(
#                     action,
#                     state_info["info"]["observation_metadata"],
#                     action_set_tag=args.action_set_tag,
#                     prompt_constructor=agent.prompt_constructor
#                     if isinstance(agent, PromptAgent)
#                     else None,
#                 )
#                 all_logs[str(config_file)].append(action_str)
                
#                 # 渲染与日志记录
#                 render_helper.render(
#                     action, state_info, meta_data, args.render_screenshot
#                 )
#                 meta_data["action_history"].append(action_str)

#                 if action["action_type"] == ActionTypes.STOP:
#                     break

#                 obs, _, terminated, _, info = env.step(action)
#                 state_info = {"observation": obs, "info": info}
#                 trajectory.append(state_info)
                
#                 with open(log_file, "w") as f:
#                     json.dump(all_logs, f, indent=4)

#                 if terminated:
#                     # add a action place holder
#                     trajectory.append(create_stop_action(""))
#                     break

#             # 评估器执行
#             evaluator = evaluator_router(config_file)
#             score = evaluator(
#                 trajectory=trajectory,
#                 config_file=config_file,
#                 page=env.page,
#                 client=env.get_page_client(env.page),
#             )

#             scores.append(score)

#             if score == 1:
#                 logger.info(f"[Result] (PASS) {config_file}")
#             else:
#                 logger.info(f"[Result] (FAIL) {config_file}")
#                 logger.info("没有任务成功完成 - scores列表为空")


#             if args.save_trace_enabled:
#                 env.save_trace(
#                     Path(args.result_dir) / "traces" / f"{task_id}.zip"
#                 )

#         except openai.error.OpenAIError as e:
#             # import traceback
#             logger.info(f"[OpenAI Error] {repr(e)}")
#             with open(Path(args.result_dir) / "error.txt", "a") as f:
#                 f.write(f"[OpenAI file]: {config_file}\n")
#                 f.write(f"[OpenAI Error] {repr(e)}\n")
#                 # f.write(traceback.format_exc())  # write stack trace to file
                
#         except Exception as e:
#             logger.info(f"[Unhandled Error] {repr(e)}]")
#             # raise e
#             # import traceback
#             # write to error file
#             with open(Path(args.result_dir) / "error.txt", "a") as f:
#                 f.write(f"[Config file]: {config_file}\n")
#                 f.write(f"[Unhandled Error] {repr(e)}\n")
#                 # f.write(traceback.format_exc())  # write stack trace to file

#         render_helper.close()

#     env.close()
#     logger.info(f"Average score: {sum(scores) / len(scores)}")



def test(
    args: argparse.Namespace,
    agent: Agent | PromptAgent | TeacherForcingAgent,
    config_file_list: list[str],
) -> None:
    scores = []
    max_steps = args.max_steps

    early_stop_thresholds = {
        "parsing_failure": args.parsing_failure_th,
        "repeating_action": args.repeating_action_failure_th,
    }

    # 增强：创建调试日志文件
    base_path = Path("/home/ubuntu/zhangzhenhao/Agent-R/AgentGym/agentenv-webarena/webarena/")
    debug_log_file = base_path / f'results/debug-{time.strftime("%Y-%m-%d-%H-%M-%S")}.log'
    with open(debug_log_file, 'w') as f:
        f.write(f"Starting test at {time.strftime('%Y-%m-%d-%H-%M-%S')}\n")
        f.flush()  # 确保写入磁盘
    
    # 环境构建
    # 增强：创建调试日志文件
    base_path = Path("/home/ubuntu/zhangzhenhao/Agent-R/AgentGym/agentenv-webarena/webarena/")
    debug_log_file = base_path / f'results/debug-{time.strftime("%Y-%m-%d-%H-%M-%S")}.log'
    with open(debug_log_file, 'w') as f:
        f.write(f"Starting test at {time.strftime('%Y-%m-%d-%H-%M-%S')}\n")
        f.flush()  # 确保写入磁盘
    
    # 环境构建
    env = ScriptBrowserEnv(
        headless=not args.render,
        slow_mo=args.slow_mo,
        observation_type=args.observation_type,
        current_viewport_only=args.current_viewport_only,
        viewport_size={
            "width": args.viewport_width,
            "height": args.viewport_height,
        },
        save_trace_enabled=args.save_trace_enabled,
        sleep_after_execution=args.sleep_after_execution,
    )
    all_logs = dict()
    log_file = base_path / f'results/log-{args.model.replace("/", "-")}-{time.strftime("%Y-%m-%d-%H-%M-%S")}.json'
    
    # 遍历任务配置文件列表
    
    # 遍历任务配置文件列表
    for config_file in config_file_list:
        try:
            # 保存原始配置文件路径
            original_config_file = config_file
            # 保存原始配置文件路径
            original_config_file = config_file
            config_file = str(base_path / config_file)
            
            with open(debug_log_file, 'a') as f:
                f.write(f"\nProcessing config file: {original_config_file}\n")
                f.write(f"Full path config file: {config_file}\n")
                f.flush()
                
            
            with open(debug_log_file, 'a') as f:
                f.write(f"\nProcessing config file: {original_config_file}\n")
                f.write(f"Full path config file: {config_file}\n")
                f.flush()
                
            render_helper = RenderHelper(
                config_file, args.result_dir, args.action_set_tag
            )

            # 初始化日志列表

            # 初始化日志列表
            all_logs[str(config_file)] = list()


            # get intent
            with open(config_file) as f:
                _c = json.load(f)
                intent = _c["intent"]
                task_id = _c["task_id"]
                
                with open(debug_log_file, 'a') as df:
                    df.write(f"Loaded config with intent: {intent}, task_id: {task_id}\n")
                    df.flush()
                
                # automatically login
                
                with open(debug_log_file, 'a') as df:
                    df.write(f"Loaded config with intent: {intent}, task_id: {task_id}\n")
                    df.flush()
                
                # automatically login
                if _c["storage_state"]:
                    cookie_file_name = os.path.basename(_c["storage_state"])
                    comb = get_site_comb_from_filepath(cookie_file_name)
                    temp_dir = tempfile.mkdtemp()
                    
                    with open(debug_log_file, 'a') as df:
                        df.write(f"Cookie info: {cookie_file_name}, temp_dir: {temp_dir}\n")
                        df.flush()
                    
                    # subprocess to renew the cookie
                    try:
                        result = subprocess.run(
                            [
                                "python",
                                "-m", "browser_env.auto_login",  # 使用模块导入方式
                                "--auth_folder",
                                temp_dir,
                                "--site_list",
                                *comb,
                            ],
                            capture_output=True,
                            text=True
                        )
                        with open(debug_log_file, 'a') as df:
                            df.write(f"Subprocess stdout: {result.stdout}\n")
                            df.write(f"Subprocess stderr: {result.stderr}\n")
                            df.flush()
                    except Exception as e:
                        with open(debug_log_file, 'a') as df:
                            df.write(f"Subprocess error: {str(e)}\n")
                            df.flush()
                    
                    _c["storage_state"] = f"{temp_dir}/{cookie_file_name}"
                    
                    with open(debug_log_file, 'a') as df:
                        df.write(f"New storage state: {_c['storage_state']}\n")
                        df.write(f"File exists: {os.path.exists(_c['storage_state'])}\n")
                        df.flush()
                    
                    # 更新配置文件
                    temp_config_file = f"{temp_dir}/{os.path.basename(config_file)}"
                    with open(temp_config_file, "w") as f:
                        json.dump(_c, f)
                        
                    # 修复：确保日志使用新路径
                    all_logs[str(temp_config_file)] = all_logs.get(str(config_file), [])
                    config_file = temp_config_file
                    
                    with open(debug_log_file, 'a') as df:
                        df.write(f"Updated config file path: {config_file}\n")
                        df.flush()
                    cookie_file_name = os.path.basename(_c["storage_state"])
                    comb = get_site_comb_from_filepath(cookie_file_name)
                    temp_dir = tempfile.mkdtemp()
                    
                    with open(debug_log_file, 'a') as df:
                        df.write(f"Cookie info: {cookie_file_name}, temp_dir: {temp_dir}\n")
                        df.flush()
                    
                    # subprocess to renew the cookie
                    try:
                        result = subprocess.run(
                            [
                                "python",
                                "-m", "browser_env.auto_login",  # 使用模块导入方式
                                "--auth_folder",
                                temp_dir,
                                "--site_list",
                                *comb,
                            ],
                            capture_output=True,
                            text=True
                        )
                        with open(debug_log_file, 'a') as df:
                            df.write(f"Subprocess stdout: {result.stdout}\n")
                            df.write(f"Subprocess stderr: {result.stderr}\n")
                            df.flush()
                    except Exception as e:
                        with open(debug_log_file, 'a') as df:
                            df.write(f"Subprocess error: {str(e)}\n")
                            df.flush()
                    
                    _c["storage_state"] = f"{temp_dir}/{cookie_file_name}"
                    
                    with open(debug_log_file, 'a') as df:
                        df.write(f"New storage state: {_c['storage_state']}\n")
                        df.write(f"File exists: {os.path.exists(_c['storage_state'])}\n")
                        df.flush()
                    
                    # 更新配置文件
                    temp_config_file = f"{temp_dir}/{os.path.basename(config_file)}"
                    with open(temp_config_file, "w") as f:
                        json.dump(_c, f)
                        
                    # 修复：确保日志使用新路径
                    all_logs[str(temp_config_file)] = all_logs.get(str(config_file), [])
                    config_file = temp_config_file
                    
                    with open(debug_log_file, 'a') as df:
                        df.write(f"Updated config file path: {config_file}\n")
                        df.flush()

            logger.info(f"[Config file]: {config_file}")
            logger.info(f"[Intent]: {intent}")

            # agent 重置并初始化
            try:
                agent.reset(config_file)
                with open(debug_log_file, 'a') as df:
                    df.write(f"Agent reset successful\n")
                    df.flush()
            except Exception as e:
                with open(debug_log_file, 'a') as df:
                    df.write(f"Agent reset error: {str(e)}\n")
                    df.flush()
                raise e

            # 交互轨迹生成
            # agent 重置并初始化
            try:
                agent.reset(config_file)
                with open(debug_log_file, 'a') as df:
                    df.write(f"Agent reset successful\n")
                    df.flush()
            except Exception as e:
                with open(debug_log_file, 'a') as df:
                    df.write(f"Agent reset error: {str(e)}\n")
                    df.flush()
                raise e

            # 交互轨迹生成
            trajectory: Trajectory = []
            try:
                obs, info = env.reset(options={"config_file": config_file})
                with open(debug_log_file, 'a') as df:
                    df.write(f"Environment reset successful\n")
                    df.flush()
                state_info: StateInfo = {"observation": obs, "info": info}
                trajectory.append(state_info)
            except Exception as e:
                with open(debug_log_file, 'a') as df:
                    df.write(f"Environment reset error: {str(e)}\n")
                    df.flush()
                raise e
            try:
                obs, info = env.reset(options={"config_file": config_file})
                with open(debug_log_file, 'a') as df:
                    df.write(f"Environment reset successful\n")
                    df.flush()
                state_info: StateInfo = {"observation": obs, "info": info}
                trajectory.append(state_info)
            except Exception as e:
                with open(debug_log_file, 'a') as df:
                    df.write(f"Environment reset error: {str(e)}\n")
                    df.flush()
                raise e

            meta_data = {"action_history": ["None"]}
            step_count = 0
            consecutive_field_inputs = 0
            last_field_id = None
            last_input_content = None
            form_submission_attempted = False
            
            step_count = 0
            consecutive_field_inputs = 0
            last_field_id = None
            last_input_content = None
            form_submission_attempted = False
            
            while True:
                step_count += 1
                with open(debug_log_file, 'a') as df:
                    df.write(f"Step {step_count} starting\n")
                    df.flush()
                    
                step_count += 1
                with open(debug_log_file, 'a') as df:
                    df.write(f"Step {step_count} starting\n")
                    df.flush()
                    
                early_stop_flag, stop_info = early_stop(
                    trajectory, max_steps, early_stop_thresholds
                )

                # early stop 判定（防止死循环）
                # early stop 判定（防止死循环）
                if early_stop_flag:
                    with open(debug_log_file, 'a') as df:
                        df.write(f"Early stop triggered: {stop_info}\n")
                        df.flush()
                    with open(debug_log_file, 'a') as df:
                        df.write(f"Early stop triggered: {stop_info}\n")
                        df.flush()
                    action = create_stop_action(f"Early stop: {stop_info}")
                else:
                    try:
                        print(f"meta data: {meta_data}")
                        print(f"meta data: {meta_data}")
                        action = agent.next_action(
                            trajectory, intent, meta_data=meta_data
                        )
                        
                        # 增强：表单提交处理
                        if action["action_type"] == ActionTypes.TYPE:
                            current_field_id = action.get("element", "")
                            current_input = action.get("value", "")
                            
                            # 检查是否重复输入同一字段
                            if current_field_id == last_field_id:
                                consecutive_field_inputs += 1
                                with open(debug_log_file, 'a') as df:
                                    df.write(f"Detected repeated form input to same field ({consecutive_field_inputs} times)\n")
                                    df.flush()
                            else:
                                consecutive_field_inputs = 0
                            
                            # 保存当前字段ID和内容
                            last_field_id = current_field_id
                            last_input_content = current_input
                            
                            # 检测是否在填写日期表单
                            if 'textbox' in str(action.get("description", "")) and ('From *' in str(action.get("description", "")) or 'To *' in str(action.get("description", ""))):
                                # 如果是"To"字段，标记为最后一个字段并添加回车
                                if 'To *' in str(action.get("description", "")):
                                    action["press_enter_after"] = 1  # 修改为按回车提交表单
                                    form_submission_attempted = True
                                    with open(debug_log_file, 'a') as df:
                                        df.write(f"Modified action to submit form with press_enter_after=1\n")
                                        df.flush()
                        
                        with open(debug_log_file, 'a') as df:
                            df.write(f"Agent action: {action['action_type']}\n")
                            df.flush()
                        
                        # 增强：表单提交处理
                        if action["action_type"] == ActionTypes.TYPE:
                            current_field_id = action.get("element", "")
                            current_input = action.get("value", "")
                            
                            # 检查是否重复输入同一字段
                            if current_field_id == last_field_id:
                                consecutive_field_inputs += 1
                                with open(debug_log_file, 'a') as df:
                                    df.write(f"Detected repeated form input to same field ({consecutive_field_inputs} times)\n")
                                    df.flush()
                            else:
                                consecutive_field_inputs = 0
                            
                            # 保存当前字段ID和内容
                            last_field_id = current_field_id
                            last_input_content = current_input
                            
                            # 检测是否在填写日期表单
                            if 'textbox' in str(action.get("description", "")) and ('From *' in str(action.get("description", "")) or 'To *' in str(action.get("description", ""))):
                                # 如果是"To"字段，标记为最后一个字段并添加回车
                                if 'To *' in str(action.get("description", "")):
                                    action["press_enter_after"] = 1  # 修改为按回车提交表单
                                    form_submission_attempted = True
                                    with open(debug_log_file, 'a') as df:
                                        df.write(f"Modified action to submit form with press_enter_after=1\n")
                                        df.flush()
                        
                        with open(debug_log_file, 'a') as df:
                            df.write(f"Agent action: {action['action_type']}\n")
                            df.flush()
                    except ValueError as e:
                        # get the error message
                        with open(debug_log_file, 'a') as df:
                            df.write(f"Agent action error: {str(e)}\n")
                            df.flush()
                        with open(debug_log_file, 'a') as df:
                            df.write(f"Agent action error: {str(e)}\n")
                            df.flush()
                        action = create_stop_action(f"ERROR: {str(e)}")

                trajectory.append(action)

                try:
                    action_str = get_action_description(
                        action,
                        state_info["info"]["observation_metadata"],
                        action_set_tag=args.action_set_tag,
                        prompt_constructor=agent.prompt_constructor
                        if isinstance(agent, PromptAgent)
                        else None,
                    )
                    with open(debug_log_file, 'a') as df:
                        df.write(f"Action description: {action_str}\n")
                        df.flush()
                    
                    # 确保日志字典有当前配置文件路径的键
                    if str(config_file) not in all_logs:
                        all_logs[str(config_file)] = []
                    all_logs[str(config_file)].append(action_str)
                    
                except Exception as e:
                    with open(debug_log_file, 'a') as df:
                        df.write(f"Action description error: {str(e)}\n")
                        df.flush()
                
                # 渲染与日志记录
                try:
                    render_helper.render(
                        action, state_info, meta_data, args.render_screenshot
                    )
                    meta_data["action_history"].append(action_str)
                except Exception as e:
                    with open(debug_log_file, 'a') as df:
                        df.write(f"Render error: {str(e)}\n")
                        df.flush()
                try:
                    action_str = get_action_description(
                        action,
                        state_info["info"]["observation_metadata"],
                        action_set_tag=args.action_set_tag,
                        prompt_constructor=agent.prompt_constructor
                        if isinstance(agent, PromptAgent)
                        else None,
                    )
                    with open(debug_log_file, 'a') as df:
                        df.write(f"Action description: {action_str}\n")
                        df.flush()
                    
                    # 确保日志字典有当前配置文件路径的键
                    if str(config_file) not in all_logs:
                        all_logs[str(config_file)] = []
                    all_logs[str(config_file)].append(action_str)
                    
                except Exception as e:
                    with open(debug_log_file, 'a') as df:
                        df.write(f"Action description error: {str(e)}\n")
                        df.flush()
                
                # 渲染与日志记录
                try:
                    render_helper.render(
                        action, state_info, meta_data, args.render_screenshot
                    )
                    meta_data["action_history"].append(action_str)
                except Exception as e:
                    with open(debug_log_file, 'a') as df:
                        df.write(f"Render error: {str(e)}\n")
                        df.flush()

                if action["action_type"] == ActionTypes.STOP:
                    with open(debug_log_file, 'a') as df:
                        df.write(f"Stop action encountered\n")
                        df.flush()
                    with open(debug_log_file, 'a') as df:
                        df.write(f"Stop action encountered\n")
                        df.flush()
                    break

                try:
                    obs, _, terminated, _, info = env.step(action)
                    state_info = {"observation": obs, "info": info}
                    trajectory.append(state_info)
                    with open(debug_log_file, 'a') as df:
                        df.write(f"Environment step successful, terminated: {terminated}\n")
                        df.flush()
                        
                    # 如果刚刚尝试提交表单，添加额外等待以确保页面加载
                    if form_submission_attempted:
                        time.sleep(2.0)  # 额外等待时间
                        form_submission_attempted = False
                        with open(debug_log_file, 'a') as df:
                            df.write(f"Added extra wait after form submission\n")
                            df.flush()
                        
                except Exception as e:
                    with open(debug_log_file, 'a') as df:
                        df.write(f"Environment step error: {str(e)}\n")
                        df.flush()
                    break
                
                # 保存日志到JSON文件
                try:
                    with open(log_file, "w") as f:
                        json.dump(all_logs, f, indent=4)
                    with open(debug_log_file, 'a') as df:
                        df.write(f"Logs saved to {log_file}\n")
                        df.flush()
                except Exception as e:
                    with open(debug_log_file, 'a') as df:
                        df.write(f"Error saving logs: {str(e)}\n")
                        df.flush()
                try:
                    obs, _, terminated, _, info = env.step(action)
                    state_info = {"observation": obs, "info": info}
                    trajectory.append(state_info)
                    with open(debug_log_file, 'a') as df:
                        df.write(f"Environment step successful, terminated: {terminated}\n")
                        df.flush()
                        
                    # 如果刚刚尝试提交表单，添加额外等待以确保页面加载
                    if form_submission_attempted:
                        time.sleep(2.0)  # 额外等待时间
                        form_submission_attempted = False
                        with open(debug_log_file, 'a') as df:
                            df.write(f"Added extra wait after form submission\n")
                            df.flush()
                        
                except Exception as e:
                    with open(debug_log_file, 'a') as df:
                        df.write(f"Environment step error: {str(e)}\n")
                        df.flush()
                    break
                
                # 保存日志到JSON文件
                try:
                    with open(log_file, "w") as f:
                        json.dump(all_logs, f, indent=4)
                    with open(debug_log_file, 'a') as df:
                        df.write(f"Logs saved to {log_file}\n")
                        df.flush()
                except Exception as e:
                    with open(debug_log_file, 'a') as df:
                        df.write(f"Error saving logs: {str(e)}\n")
                        df.flush()

                if terminated:
                    # add a action place holder
                    trajectory.append(create_stop_action(""))
                    with open(debug_log_file, 'a') as df:
                        df.write(f"Environment terminated\n")
                        df.flush()
                    with open(debug_log_file, 'a') as df:
                        df.write(f"Environment terminated\n")
                        df.flush()
                    break

            # 评估器执行
            try:
                evaluator = evaluator_router(config_file)
                score = evaluator(
                    trajectory=trajectory,
                    config_file=config_file,
                    page=env.page,
                    client=env.get_page_client(env.page),
                )
                scores.append(score)
                with open(debug_log_file, 'a') as df:
                    df.write(f"Evaluation completed with score: {score}\n")
                    df.flush()
            # 评估器执行
            try:
                evaluator = evaluator_router(config_file)
                score = evaluator(
                    trajectory=trajectory,
                    config_file=config_file,
                    page=env.page,
                    client=env.get_page_client(env.page),
                )
                scores.append(score)
                with open(debug_log_file, 'a') as df:
                    df.write(f"Evaluation completed with score: {score}\n")
                    df.flush()

                if score == 1:
                    logger.info(f"[Result] (PASS) {config_file}")
                else:
                    logger.info(f"[Result] (FAIL) {config_file}")
            except Exception as e:
                with open(debug_log_file, 'a') as df:
                    df.write(f"Evaluation error: {str(e)}\n")
                    df.flush()
                logger.info(f"[Evaluation Error] {repr(e)}")
                if score == 1:
                    logger.info(f"[Result] (PASS) {config_file}")
                else:
                    logger.info(f"[Result] (FAIL) {config_file}")
            except Exception as e:
                with open(debug_log_file, 'a') as df:
                    df.write(f"Evaluation error: {str(e)}\n")
                    df.flush()
                logger.info(f"[Evaluation Error] {repr(e)}")

            if args.save_trace_enabled:
                try:
                    env.save_trace(
                        Path(args.result_dir) / "traces" / f"{task_id}.zip"
                    )
                    with open(debug_log_file, 'a') as df:
                        df.write(f"Trace saved\n")
                        df.flush()
                except Exception as e:
                    with open(debug_log_file, 'a') as df:
                        df.write(f"Error saving trace: {str(e)}\n")
                        df.flush()
                try:
                    env.save_trace(
                        Path(args.result_dir) / "traces" / f"{task_id}.zip"
                    )
                    with open(debug_log_file, 'a') as df:
                        df.write(f"Trace saved\n")
                        df.flush()
                except Exception as e:
                    with open(debug_log_file, 'a') as df:
                        df.write(f"Error saving trace: {str(e)}\n")
                        df.flush()

        except openai.error.OpenAIError as e:
            logger.info(f"[OpenAI Error] {repr(e)}")
            with open(debug_log_file, 'a') as df:
                df.write(f"OpenAI error: {str(e)}\n")
                df.flush()
            with open(debug_log_file, 'a') as df:
                df.write(f"OpenAI error: {str(e)}\n")
                df.flush()
            with open(Path(args.result_dir) / "error.txt", "a") as f:
                f.write(f"[OpenAI file]: {config_file}\n")
                f.write(f"[OpenAI Error] {repr(e)}\n")
                
        except Exception as e:
            logger.info(f"[Unhandled Error] {repr(e)}]")
            with open(debug_log_file, 'a') as df:
                df.write(f"Unhandled error: {str(e)}\n")
                df.flush()
            with open(debug_log_file, 'a') as df:
                df.write(f"Unhandled error: {str(e)}\n")
                df.flush()
            with open(Path(args.result_dir) / "error.txt", "a") as f:
                f.write(f"[Config file]: {config_file}\n")
                f.write(f"[Unhandled Error] {repr(e)}\n")

        try:
            render_helper.close()
            with open(debug_log_file, 'a') as df:
                df.write(f"Render helper closed\n")
                df.flush()
        except Exception as e:
            with open(debug_log_file, 'a') as df:
                df.write(f"Error closing render helper: {str(e)}\n")
                df.flush()
        try:
            render_helper.close()
            with open(debug_log_file, 'a') as df:
                df.write(f"Render helper closed\n")
                df.flush()
        except Exception as e:
            with open(debug_log_file, 'a') as df:
                df.write(f"Error closing render helper: {str(e)}\n")
                df.flush()

    env.close()
    
    # 修复除零错误
    if scores:
        logger.info(f"Average score: {sum(scores) / len(scores)}")
    else:
        logger.info("No successful tests - scores list is empty")
        
    # 关闭日志处理程序
    for handler in logger.handlers:
        handler.flush()



# 运行前准备
# 准备函数，转换提示文件、准备结果目录、记录日志文件。
    
    # 修复除零错误
    if scores:
        logger.info(f"Average score: {sum(scores) / len(scores)}")
    else:
        logger.info("No successful tests - scores list is empty")
        
    # 关闭日志处理程序
    for handler in logger.handlers:
        handler.flush()



# 运行前准备
# 准备函数，转换提示文件、准备结果目录、记录日志文件。
def prepare(args: argparse.Namespace) -> None:
    # convert prompt python files to json
    from agent.prompts import to_json

    to_json.run()

    # prepare result dir
    result_dir = args.result_dir
    if not result_dir:
        result_dir = (
            f"cache/results_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
        )
    if not Path(result_dir).exists():
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        args.result_dir = result_dir
        logger.info(f"Create result dir: {result_dir}")

    if not (Path(result_dir) / "traces").exists():
        (Path(result_dir) / "traces").mkdir(parents=True)

    # log the log file
    with open(os.path.join(result_dir, "log_files.txt"), "a+") as f:
        f.write(f"{LOG_FILE_NAME}\n")


# 获取未完成任务的函数，通过比较配置文件和结果文件确定未完成的任务。
# 获取未完成任务的函数，通过比较配置文件和结果文件确定未完成的任务。
def get_unfinished(config_files: list[str], result_dir: str) -> list[str]:
    result_files = glob.glob(f"{result_dir}/*.html")
    task_ids = [
        os.path.basename(f).split(".")[0].split("_")[1] for f in result_files
    ]
    unfinished_configs = []
    for config_file in config_files:
        task_id = os.path.basename(config_file).split(".")[0]
        if task_id not in task_ids:
            unfinished_configs.append(config_file)
    return unfinished_configs


def dump_config(args: argparse.Namespace) -> None:
    config_file = Path(args.result_dir) / "config.json"
    if not config_file.exists():
        with open(config_file, "w") as f:
            json.dump(vars(args), f, indent=4)
            logger.info(f"Dump config to {config_file}")


if __name__ == "__main__":
    args = config()
    # sleep_after_execution: 加到3.0（确保页面加载完成）
    args.sleep_after_execution = 3.0
    # sleep_after_execution: 加到3.0（确保页面加载完成）
    args.sleep_after_execution = 3.0
    prepare(args)

    test_file_list = []
    st_idx = args.test_start_idx
    ed_idx = args.test_end_idx
    for i in range(st_idx, ed_idx):
        test_file_list.append(f"config_files/{i}.json")
    # if "debug" not in args.result_dir:
    #     test_file_list = get_unfinished(test_file_list, args.result_dir)

    if len(test_file_list) == 0:
        logger.info("No task left to run")
    else:
        print(f"Total {len(test_file_list)} tasks left")
        args.render = False
        args.render_screenshot = True
        args.save_trace_enabled = True

        args.current_viewport_only = True
        dump_config(args)

        agent = construct_agent(args)
        test(args, agent, test_file_list)
