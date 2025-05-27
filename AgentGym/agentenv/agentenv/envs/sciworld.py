from typing import Any, Mapping
import requests
from requests.exceptions import RequestException
from AgentGym.agentenv.agentenv.controller import BaseEnvClient, BaseTask, ConversationMessage, StepOutput


class SciworldEnvClient(BaseEnvClient):
    conversation_start = (
        ConversationMessage(
            {
                "from": "human",
                "loss": None,
                "value": 'You are an agent for science world. Every round I will give you an observation, you have to respond an action based on the observation to finish the given task. Here are the actions you may take: [{"action": "open/close OBJ", "description": "open/close a container"}, {"action": "de/activate OBJ", "description": "activate/deactivate a device"}, {"action": "connect OBJ to OBJ", "description": "connect electrical components"}, {"action": "disconnect OBJ", "description": "disconnect electrical components"}, {"action": "use OBJ [on OBJ]", "description": "use a device/item"}, {"action": "look around", "description": "describe the current room"}, {"action": "look at OBJ", "description": "describe an object in detail"}, {"action": "look in OBJ", "description": "describe a container\'s contents"}, {"action": "read OBJ", "description": "read a note or book"}, {"action": "move OBJ to OBJ", "description": "move an object to a container"}, {"action": "pick up OBJ", "description": "move an object to the inventory"}, {"action": "put down OBJ", "description": "drop an inventory item"}, {"action": "pour OBJ into OBJ", "description": "pour a liquid into a container"}, {"action": "dunk OBJ into OBJ", "description": "dunk a container into a liquid"}, {"action": "mix OBJ", "description": "chemically mix a container"}, {"action": "go to LOC", "description": "move to a new location"}, {"action": "eat OBJ", "description": "eat a food"}, {"action": "flush OBJ", "description": "flush a toilet"}, {"action": "focus on OBJ", "description": "signal intent on a task object"}, {"action": "wait", "description": "take no action for 10 iterations"}, {"action": "wait1", "description": "take no action for 1 iteration"}, {"action": "task", "description": "describe current task"}, {"action": "inventory", "description": "list your inventory"}]\nYour response should use the following format:\nThought:\nyour thoughts.\n\nAction:\nyour next action',
            }
        ),
        ConversationMessage(
            {
                "from": "gpt",
                "loss": False,
                "value": "OK. I'll follow your instructions and try my best to solve the task.",
            }
        ),
    )

    def __init__(
        self, env_server_base: str, data_len: int, *args, timeout: int = 300, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len

        ok = requests.post(f"{self.env_server_base}/create", timeout=self.timeout)
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")

        ok = ok.json()
        self.env_id = ok["id"]
        print("Current Env", self.env_id)

    def __len__(self):
        return self.data_len

    def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        data["id"] = self.env_id
        res = requests.post(
            f"{self.env_server_base}/{path}",
            json=data,
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def _get(self, path: str) -> dict[str, Any]:
        res = requests.get(
            f"{self.env_server_base}/{path}?id={self.env_id}",
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def reset(self, data_idx: int = 0) -> dict[str, Any]:
        response = self._post("reset", {"data_idx": data_idx})
        self.info = {
            "observation": response["task_description"] + '\n' + response["observation"],
            "reward": 0,
            "score": 0,
            "done": False,
        }
        return response


    def step(self, action: str) -> StepOutput:
        if action.endswith("</s>"):
            action = action[:-5]
        _action = action.split("Action:")
        if len(_action) > 1:
            action = _action[1].strip()
        else:
            action = _action[0].strip()
        response = self._post("step", {"action": action})
        self.info = {
            "observation": response["observation"],
            "reward": response["reward"],
            "score": response["score"],
            "done": response["done"],
        }
        return StepOutput(
            state=response["observation"],
            reward=response["score"],
            done=response["done"],
        )

    def observe(self) -> str:
        return self.info["observation"]

    # new implement 
    def get_look_around(self) -> str:
        look_around = self._get("look_around")
        if "look_around" in look_around:
            look_around = look_around["look_around"]
        else:
            look_around = ""
        return look_around

    #  def get_golden_action_seq(self) -> str:
    #     golden_path = self._get("golden_action_seq")['golden_action_seq']
    #     return golden_path
    

    def get_valid_action_object_combinations(self) -> str:
        combinations = self._get("valid_action_object_combinations")
        if "get_valid_action_object_combinations" in combinations:
            combinations = combinations["get_valid_action_object_combinations"]
        else:
            combinations = ""
        return combinations


    def get_inventory(self) -> str:
        inventory = self._get("inventory")
        if "inventory" in inventory:
            inventory = inventory["inventory"]
        else:
            inventory = ""
        return inventory
    
    def get_game_nums(self) -> int:
        return self._get("game_nums")



    # def get_look_around(self):
    #     result = self.step("look around")
    #     print("🔍 StepOutput result =", result)
    #     print("🔍 StepOutput dir =", dir(result))             
    #     return result
   
    # def get_inventory(self):
    #     result = self.step("inventory")
    #     print("🎒 StepOutput (inventory) =", result)
    #     return result.state  # 因为 StepOutput 的 observation 对应的是 state


    # def get_valid_action_object_combinations(self):
    #     print("[WARN] valid_action_object_combinations() not implemented on server, returning []")
    #     return []



class SciworldTask(BaseTask):
    env_client_cls = SciworldEnvClient
    env_name = "SciWorld"

    def __init__(
        self, client_args: Mapping[str, Any], *args, n_clients: int = 1, **kwargs
    ) -> None:
        super().__init__(client_args, n_clients, *args, **kwargs)
