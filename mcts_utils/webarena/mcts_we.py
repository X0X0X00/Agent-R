"""
WebArena MCTS 推理模块
"""

import os
import re
from copy import deepcopy
from dataclasses import dataclass
import mmengine
import warnings
from fastchat.model.model_adapter import get_conversation_template
from agentenv.envs import WebarenaEnvClient
from mcts_utils.mcts_raw import MCTSNode, MCTSAgent
from mcts_utils.llm_server import FuncCall

warnings.simplefilter("ignore", DeprecationWarning)

@dataclass
class MCTSConfig:
    max_depth: int = int(os.environ.get("MAX_DEPTH", 10))
    iterations: int = int(os.environ.get("ITERA", 4))
    n_generate_samples: int = int(os.environ.get("N_GEN", 4))
    coef: float = 0.25


class ExtendedNode(MCTSNode):
    def __init__(self, 
                 env=None,
                 recent_actions=None,
                 action="",
                 obs="",
                 disaster=False,
                 env_score=0,
                 puct_value=0,
                 **kwargs):
        super().__init__(**kwargs)
        self.env = env
        self.env_score = env_score
        self.recent_actions = recent_actions or []
        self.action = action
        self.disaster = disaster
        self.puct_value = puct_value
        self.obs = obs

    @property
    def reward(self):
        return self.env_score

    def to_dict(self):
        return {
            'visits': self.visits,
            'value': self.value,
            'prior': self.prior,
            'puct_value': self.puct,
            'obs': self.obs,
            'llm_response': self.llm_response,
            'depth': self.depth,
            'is_terminal': self.is_terminal,
            'recent_actions': self.recent_actions,
            'action': self.action,
            'env_score': self.env_score,
            'disaster': self.disaster,
            'state': self.state.to_openai_api_messages(),
            'children': [child.to_dict() for child in self.children]
        }


class ExtendedMCTS(MCTSAgent):
    def __init__(self, idx=0, calling=None, encoding=None, max_len=8000, model_name=None, env=None, generate_cfg=MCTSConfig()):
        super().__init__()
        self.generate_cfg = generate_cfg
        self.calling = calling
        self.encoding = encoding
        self.max_len = max_len
        self.model_name = model_name
        self.env = env
        self.idx = idx

    def search(self, env, conv, recent_actions):
        env_reward, env_done = 0, False
        recent_actions_temp = []

        for agent_response in recent_actions:
            action = agent_response.split("Action:")[-1].strip()
            conv.append_message(conv.roles[1], None)
            step_output = self.env.step(action)
            env_state, env_reward, env_done = step_output.state, step_output.reward, step_output.done
            current_obs = env.observe()
            recent_actions_temp.append([action, current_obs])
            conv.update_last_message(agent_response)
            conv.append_message(conv.roles[0], current_obs)

        init_state = deepcopy(conv)
        self.root = ExtendedNode(env=env, state=init_state, llm_response="ROOT", is_terminal=env_done,
                                 recent_actions=recent_actions_temp, env_score=env_reward, action="ROOT")

        for iter in range(self.generate_cfg.iterations):
            node = self.root
            if node.is_terminal:
                print(f"[MCTS] Stop at Iter {iter}")
                return
            while node and not node.is_terminal:
                self.expand(node)
                node = self._select(node)
        return

    def _generate(self, node):
        conv = deepcopy(node.state)
        while len(self.calling.encoding.encode(str(conv))) > self.max_len - 60:
            del conv.messages[4:6]

        prompt = conv.to_openai_api_messages()
        agent_response = self.calling.llm_func(prompt, self.model_name)
        conv.append_message(conv.roles[1], None)
        conv.update_last_message(agent_response)

        _ = self.env.reset(self.idx)
        for action in node.recent_actions:
            _ = self.env.step(action[0])

        new_action = self._extract_action(agent_response)
        step_output = self.env.step(new_action)
        current_obs = self.clean(step_output.state)
        new_env_score = step_output.reward
        is_terminal = step_output.done or new_env_score < 0

        conv.append_message(conv.roles[0], current_obs)
        new_recent_actions = deepcopy(node.recent_actions)
        new_recent_actions.append([new_action, current_obs])

        return ExtendedNode(
            obs=current_obs,
            action=new_action,
            env=self.env,
            state=conv,
            parent=node,
            disaster=new_env_score < 0,
            recent_actions=new_recent_actions,
            llm_response=agent_response,
            depth=node.depth + 1,
            env_score=new_env_score,
            is_terminal=node.depth + 1 > self.generate_cfg.max_depth or is_terminal
        )

    def _extract_action(self, response):
        if "Action:" in response:
            return re.split(r"Action:\s*", response)[-1].strip()
        return response.strip().split("\n")[-1]

    def expand(self, node):
        if not node.is_fully_expanded:
            sampled_nodes = [self._generate(node) for _ in range(self.generate_cfg.n_generate_samples)]
            dedup_nodes, fingerprint = [], set()
            for sample_node in sampled_nodes:
                if sample_node.llm_response not in fingerprint:
                    fingerprint.add(sample_node.llm_response)
                    dedup_nodes.append(sample_node)
            node.children = dedup_nodes
            for child in node.children:
                if child.is_terminal:
                    self._backpropagate(child, child.reward)

    def clean(self, s):
        return s.replace('\n', ' ').replace('\t', ' ')


# 用于 eval.py 的 entry（如果需要）
def initialize_environment(env_server_base: str, data_len: int = 200):
    return WebarenaEnvClient(env_server_base, data_len)

def setup_conversation(env):
    conv = get_conversation_template("gpt-4")
    conv.append_message(conv.roles[0], env.conversation_start[0]["value"])
    conv.append_message(conv.roles[1], "Ok.")
    observation = env.observe()
    conv.append_message(conv.roles[0], observation)
    return conv
