from typing import Any, Mapping, Dict
import json
import os
import tempfile
import subprocess

from AgentGym.agentenv.agentenv.controller import BaseEnvClient, BaseTask, ConversationMessage, StepOutput

# Import webarena components
from browser_env import ScriptBrowserEnv, Trajectory, StateInfo
from browser_env.auto_login import get_site_comb_from_filepath


class WebarenaEnvClient(BaseEnvClient):
    conversation_start = (
        ConversationMessage(
            {
                "from": "human",
                "loss": None,
                "value": "You are an autonomous intelligent agent tasked with navigating a web browser. You will be given web-based tasks. These tasks will be accomplished through the use of specific actions you can issue.\n\nHere's the information you'll have:\nThe user's objective: This is the task you're trying to complete.\nThe current web page's accessibility tree: This is a simplified representation of the webpage, providing key information.\nThe current web page's URL: This is the page you're currently navigating.\nThe open tabs: These are the tabs you have open.\nThe previous action: This is the action you just performed. It may be helpful to track your progress.\n\nThe actions you can perform fall into several categories:\n\nPage Operation Actions:\n`click [id]`: This action clicks on an element with a specific id on the webpage.\n`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the \"Enter\" key is pressed after typing unless press_enter_after is set to 0.\n`hover [id]`: Hover over an element with id.\n`press [key_comb]`:  Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).\n`scroll [direction=down|up]`: Scroll the page up or down.\n\nTab Management Actions:\n`new_tab`: Open a new, empty browser tab.\n`tab_focus [tab_index]`: Switch the browser's focus to a specific tab using its index.\n`close_tab`: Close the currently active tab.\n\nURL Navigation Actions:\n`goto [url]`: Navigate to a specific URL.\n`go_back`: Navigate to the previously viewed page.\n`go_forward`: Navigate to the next page (if a previous 'go_back' action was performed).\n\nCompletion Action:\n`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as \"N/A\" in the bracket.\n\nHomepage:\nIf you want to visit other websites, check out the homepage at http://homepage.com. It has a list of websites you can visit.\nhttp://homepage.com/password.html lists all the account name and password for the websites. You can use them to log in to the websites.\n\nTo be successful, it is very important to follow the following rules:\n1. You should only issue an action that is valid given the current observation\n2. You should only issue one action at a time.\n3. You should follow the examples to reason step by step and then issue the next action.\n4. Generate the action in the correct format. Start with a \"In summary, the next action I will perform is\" phrase, followed by action inside ``````. For example, \"In summary, the next action I will perform is ```click [1234]```\".\n5. Issue stop action when you think you have achieved the objective. Don't generate anything after stop.",
            }
        ),
        ConversationMessage({"from": "gpt", "loss": False, "value": "Ok."}),
    )

    def __init__(
        self, 
        data_len: int,
        config_file_dir: str = "/home/ubuntu/zhangzhenhao/Agent-R/AgentGym/agentenv-webarena/webarena/config_files",
        headless: bool = True,
        slow_mo: int = 0,  
        observation_type: str = "accessibility_tree",
        current_viewport_only: bool = True,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        save_trace_enabled: bool = False,
        sleep_after_execution: float = 2.0,
        *args, 
        **kwargs
    ):

        # 尝试不传参数
        super().__init__()
        # super().__init__(*args, **kwargs)
        self.data_len = data_len
        self.config_file_dir = config_file_dir
        
        # Initialize ScriptBrowserEnv directly
        self.env = ScriptBrowserEnv(
            headless=headless,
            slow_mo=slow_mo,
            observation_type=observation_type,
            current_viewport_only=current_viewport_only,
            viewport_size={
                "width": viewport_width,
                "height": viewport_height,
            },
            save_trace_enabled=save_trace_enabled,
            sleep_after_execution=sleep_after_execution,
        )
        
        # Store current state info for setup_conversation compatibility
        self.info = {"observation": ""}
        self._current_observation = ""
        self._current_config_file = None
        self._task_intent = ""

    def __len__(self):
        return self.data_len

    def _load_config_file(self, idx: int) -> str:
        """Load and process config file for the given task index."""
        config_file = f"{self.config_file_dir}/{idx}.json"
        
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_file) as f:
            config = json.load(f)
            
        # Handle automatic login if required
        if config.get("storage_state"):
            cookie_file_name = os.path.basename(config["storage_state"])
            comb = get_site_comb_from_filepath(cookie_file_name)
            temp_dir = tempfile.mkdtemp()
            
            # Subprocess to renew the cookie
            subprocess.run(
                [
                    "python",
                    # "browser_env/auto_login.py",
                    "/home/ubuntu/zhangzhenhao/Agent-R/AgentGym/agentenv-webarena/webarena/browser_env/auto_login.py",
                    "--auth_folder",
                    temp_dir,
                    "--site_list",
                    *comb,
                ],
                check=False  # Don't raise exception if subprocess fails
            )
            
            cookie_path = f"{temp_dir}/{cookie_file_name}"
            if os.path.exists(cookie_path):
                config["storage_state"] = cookie_path
            
            # Update the config file
            updated_config_file = f"{temp_dir}/{os.path.basename(config_file)}"
            with open(updated_config_file, "w") as f:
                json.dump(config, f)
            config_file = updated_config_file
        
        return config_file

    def observe(self) -> str:
        """Get the current observation from the environment.
        
        Returns:
            str: The observation string that can be directly used in conversation.
        """
        try:
            # Get current observation from browser env
            # Note: ScriptBrowserEnv doesn't have a direct observe() method
            # The observation is typically obtained after reset() or step()
            if hasattr(self.env, '_get_obs'):
                obs = self.env._get_obs()
            else:
                # If we can't get observation directly, return cached observation
                obs = self._current_observation
            
            if isinstance(obs, dict):
                # Format the observation for conversation
                obs_parts = []
                if "url" in obs:
                    obs_parts.append(f"Current URL: {obs['url']}")
                if "page_title" in obs:
                    obs_parts.append(f"Page Title: {obs['page_title']}")
                if "accessibility_tree" in obs:
                    obs_parts.append(f"Accessibility Tree:\n{obs['accessibility_tree']}")
                elif "text" in obs:
                    obs_parts.append(f"Page Content:\n{obs['text']}")
                if "open_tabs" in obs:
                    obs_parts.append(f"Open Tabs: {obs['open_tabs']}")
                if "previous_action" in obs:
                    obs_parts.append(f"Previous Action: {obs['previous_action']}")
                
                formatted_obs = "\n\n".join(obs_parts) if obs_parts else str(obs)
                self._current_observation = formatted_obs
                return formatted_obs
            else:
                self._current_observation = str(obs)
                return str(obs)
                
        except Exception as e:
            error_msg = f"Error getting observation: {e}"
            self._current_observation = error_msg
            return error_msg

    def step(self, action: str) -> StepOutput:
        """Execute an action in the environment and return the result."""
        try:
            # Parse action from the LLM output
            # ScriptBrowserEnv expects action dict, need to parse the action string
            from browser_env.actions import create_id_based_action, create_playwright_action, ActionParsingError
            
            # Try to parse the action
            try:
                # Extract action from the formatted string (e.g., "In summary, the next action I will perform is ```click [1234]```")
                if "```" in action:
                    # Extract content between ``` marks
                    action_content = action.split("```")[1].strip()
                else:
                    action_content = action.strip()
                
                # Create action dict
                parsed_action = create_id_based_action(action_content)
                
            except (ActionParsingError, IndexError) as e:
                # If parsing fails, create a NONE action
                from browser_env.actions import create_none_action
                parsed_action = create_none_action()
                parsed_action["raw_prediction"] = action
            
            # Execute action in browser environment
            obs, reward, terminated, truncated, info = self.env.step(parsed_action)
            
            # Update current observation and info
            self._current_observation = obs
            self.info = {"observation": obs, **info}
            
            return StepOutput(
                state=obs,
                reward=reward if reward is not None else 0,
                done=terminated or truncated,
            )
            
        except Exception as e:
            print(f"Error in step: {e}")
            return StepOutput(
                state={"error": str(e)},
                reward=0,
                done=True,
            )

    def reset(self, idx: int) -> Dict[str, Any]:
        """Reset the environment to a new task specified by idx.
        
        Args:
            idx: The index of the task to reset to.
            
        Returns:
            Initial observation and info from the reset environment.
        """
        try:
            # Load config file for this task
            config_file = self._load_config_file(idx)
            self._current_config_file = config_file
            
            # Get task intent from config
            with open(config_file) as f:
                config = json.load(f)
                self._task_intent = config.get("intent", "")
            
            # Reset browser environment with config
            obs, info = self.env.reset(options={"config_file": config_file})
            
            # Update internal state
            self._current_observation = obs
            self.info = {"observation": obs, **info}
            
            return {
                "observation": obs,
                "info": info
            }
            
        except Exception as e:
            print(f"Error in reset: {e}")
            # Return error state
            error_info = {"error": str(e)}
            self.info = {"observation": error_info}
            return {
                "observation": error_info,
                "info": error_info
            }
    
    def get_task_intent(self) -> str:
        """Get the current task intent/objective."""
        return self._task_intent
    
    def close(self):
        """Close the environment and free resources."""
        try:
            self.env.close()
            print("ScriptBrowserEnv closed successfully")
        except Exception as e:
            print(f"Error closing ScriptBrowserEnv: {e}")
    
    def __del__(self):
        """Destructor to ensure environment is closed when object is deleted."""
        try:
            self.close()
        except:
            pass  # Ignore errors during cleanup


class WebarenaTask(BaseTask):
    env_client_cls = WebarenaEnvClient
    env_name = "Webarena"

    def __init__(
        self,
        client_args: Mapping[str, Any],
        n_clients: int,
        *args,
        **kwargs,
    ):
        super().__init__(client_args, n_clients, *args, **kwargs)
        
    def get_env_client(self, idx: int) -> WebarenaEnvClient:
        """Get the environment client for the specified index."""
        return self.env_clients[idx % len(self.env_clients)]


'''
原版
from typing import Any, Mapping, Dict

import requests
from requests.exceptions import RequestException

from agentenv.controller import BaseEnvClient, BaseTask, ConversationMessage, StepOutput


class WebarenaEnvClient(BaseEnvClient):
    conversation_start = (
        ConversationMessage(
            {
                "from": "human",
                "loss": None,
                "value": "You are an autonomous intelligent agent tasked with navigating a web browser. You will be given web-based tasks. These tasks will be accomplished through the use of specific actions you can issue.\n\nHere's the information you'll have:\nThe user's objective: This is the task you're trying to complete.\nThe current web page's accessibility tree: This is a simplified representation of the webpage, providing key information.\nThe current web page's URL: This is the page you're currently navigating.\nThe open tabs: These are the tabs you have open.\nThe previous action: This is the action you just performed. It may be helpful to track your progress.\n\nThe actions you can perform fall into several categories:\n\nPage Operation Actions:\n`click [id]`: This action clicks on an element with a specific id on the webpage.\n`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the \"Enter\" key is pressed after typing unless press_enter_after is set to 0.\n`hover [id]`: Hover over an element with id.\n`press [key_comb]`:  Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).\n`scroll [direction=down|up]`: Scroll the page up or down.\n\nTab Management Actions:\n`new_tab`: Open a new, empty browser tab.\n`tab_focus [tab_index]`: Switch the browser's focus to a specific tab using its index.\n`close_tab`: Close the currently active tab.\n\nURL Navigation Actions:\n`goto [url]`: Navigate to a specific URL.\n`go_back`: Navigate to the previously viewed page.\n`go_forward`: Navigate to the next page (if a previous 'go_back' action was performed).\n\nCompletion Action:\n`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as \"N/A\" in the bracket.\n\nHomepage:\nIf you want to visit other websites, check out the homepage at http://homepage.com. It has a list of websites you can visit.\nhttp://homepage.com/password.html lists all the account name and password for the websites. You can use them to log in to the websites.\n\nTo be successful, it is very important to follow the following rules:\n1. You should only issue an action that is valid given the current observation\n2. You should only issue one action at a time.\n3. You should follow the examples to reason step by step and then issue the next action.\n4. Generate the action in the correct format. Start with a \"In summary, the next action I will perform is\" phrase, followed by action inside ``````. For example, \"In summary, the next action I will perform is ```click [1234]```\".\n5. Issue stop action when you think you have achieved the objective. Don't generate anything after stop.",
            }
        ),
        ConversationMessage({"from": "gpt", "loss": False, "value": "Ok."}),
    )

    # 初始化一个env
    def __init__(
        self, env_server_base: str, data_len: int, *args, timeout: int = 300, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len

        # ok = requests.post(
        #     f"{self.env_server_base}/create",
        #     timeout=self.timeout,
        # )
        
        # if ok.status_code != 200:
        #     raise RequestException(f"Failed to create environment: {ok}")

        # self.env_id = ok.json()["env_idx"]

        self.env_id = 0

    def __len__(self):
        return self.data_len


    def _post(self, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        data["env_idx"] = self.env_id
        res = requests.post(
            f"{self.env_server_base}/{path}",
            json=data,
            timeout=self.timeout,
        )
        assert res.status_code == 200, f"POST {path} failed: {res.text}"
        return res.json()

    def _get(self, path: str) -> Dict[str, Any]:
        res = requests.get(
            f"{self.env_server_base}/{path}?env_idx={self.env_id}",
            timeout=self.timeout,
        )
        assert res.status_code == 200, f"POST {path} failed: {res.text}"
        return res.json()

    def observe(self) -> Dict[str, Any]:
        response = self._get("observation")
        return response

    # run.py next action
    def step(self, action: str) -> StepOutput:
        # action is the original output of llm
        response = self._post("step", {"action": action})
        return StepOutput(
            state=response["observation"],
            reward=response["reward"],
            done=response["terminated"],
        )


    def reset(self, idx: int) -> Dict[str, Any]:
        pass


class WebarenaTask(BaseTask):
    env_client_cls = WebarenaEnvClient
    env_name = "Webarena"

    def __init__(
        self,
        client_args: Mapping[str, Any] | Mapping[str, Any],
        n_clients: int,
        *args,
        **kwargs,
    ):
        super().__init__(client_args, n_clients, *args, **kwargs)
'''