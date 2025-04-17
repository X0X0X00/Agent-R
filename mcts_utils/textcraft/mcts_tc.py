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

from copy import deepcopy
from dataclasses import dataclass
import mmengine
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
from mcts_utils.mcts_raw import MCTSNode, MCTSAgent
import os
import re
# from utils.logger import logger

from utils.logger import logger  # If it's in a utils package

@dataclass
class MCTSConfig:
    max_depth: int = int(os.environ["MAX_DEPTH"])
    iterations: int = int(os.environ["ITERA"])
    n_generate_samples: int = int(os.environ["N_GEN"])
    coef = 0.25


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
        self.recent_actions = recent_actions
        self.action = action
        self.disaster = disaster
        self.puct_value = puct_value
        self.obs = obs

        logger.info("[INIT] MCTS ExtendedNode initialized.")

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
    def __init__(self, 
                 idx=0,
                 calling=None,
                 encoding=None,
                 max_len=0,
                 model_name=None,
                 env=None,
                 generate_cfg=MCTSConfig()):
        super().__init__()
        self.generate_cfg = generate_cfg
        self.calling = calling
        self.encoding = encoding
        self.max_len = max_len
        self.model_name = model_name
        self.env = env
        self.idx = idx

    def search(self, env, conv, recent_actions):
        env_score = 0
        recent_actions_temp = []
        env_reward = 0
        env_done = False
        for agent_response in recent_actions:
            action = agent_response.split("Action:")[-1].strip()
            conv.append_message(conv.roles[1], None)
            step_output = self.env.step(action)
            env_state, env_reward, env_done = (
                step_output.state,
                step_output.reward,
                step_output.done,
            )
            current_obs = env.observe()
            recent_actions_temp.append([action, current_obs])
            conv.update_last_message(agent_response)
            conv.append_message(conv.roles[0], current_obs)

        init_state = deepcopy(conv)
        env_score = env_reward
        is_terminal = env_done
        self.root = ExtendedNode(
            env=env,
            state=init_state,
            llm_response="ROOT",
            is_terminal=is_terminal,
            recent_actions=recent_actions_temp,
            env_score=env_score,
            action="ROOT"
        )

        logger.info(f"[ROOT] MCTS root initialized. Terminal={is_terminal}, Score={env_score}")

        for iter in range(self.generate_cfg.iterations):
            node = self.root
            print(f"[MCTS] Iteration {iter+1}/{self.generate_cfg.iterations}")
            if node.is_terminal:
                logger.info(f"[MCTS] Iteration {iter} — terminal node")
                return
            while node and not node.is_terminal:
                self.expand(node)
                node = self._select(node)
        return

    def _generate(self, node):
        logger.info("[MCTS] Entered _generate()")
        conv = deepcopy(node.state)

        # 清理超长 prompt
        while len(self.calling.encoding.encode(str(conv))) > self.max_len - 60:
            del conv.messages[4:6]
            if len(conv.messages) > 4 and conv.messages[4][1].startswith('The preceding task has ended.'):
                del conv.messages[2:4]

        prompt = conv.to_openai_api_messages()
        token_len = len(self.calling.encoding.encode(str(prompt)))
        logger.debug(f"[MCTS] Prompt token length: {token_len}")

        agent_response = self.calling.llm_func(prompt, self.model_name)

        # agent_response = "Thought: Let's try.\nAction: craft 1 stone_button"


        if not agent_response or len(agent_response.strip()) == 0:
            logger.warning("[MCTS] Model returned empty response")
            return None
        logger.debug(f"[GEN] LLM Response: {agent_response}")

        # 替换动作格式
        conv = deepcopy(node.state)
        conv.append_message(conv.roles[1], None)
        conv.update_last_message(
            agent_response.replace("Action 1:", "Action:").replace("Thought 1:", "Thought:")
        )

        _ = self.env.reset(self.idx)
        for action, _ in node.recent_actions:
            _ = self.env.step(action)

        if "Action 1:" in agent_response:
            new_action = agent_response.split("Action 1:")[-1].strip()
        elif "Action:" in agent_response:
            new_action = agent_response.split("Action:")[-1].strip()
        else:
            new_action = agent_response.split('\n')[-1].strip()
        new_action = re.sub(r'^\d+\.\s*', '', new_action)

        logger.debug(f"[ACT] New action: {new_action}")

        try:
            step_output = self.env.step(new_action)
        except Exception as e:
            logger.error(f"[ERROR] env.step failed: {e}")
            return None

        current_obs = self.clean(step_output.state)
        new_env_score = step_output.reward
        is_terminal = step_output.done
        disaster = new_env_score < 0

        if disaster:
            new_env_score = 0
            is_terminal = True

        logger.debug(f"[OBS] {current_obs}")
        logger.debug(f"[REWARD] {new_env_score}, Terminal={is_terminal}")

        current_recent_actions = deepcopy(node.recent_actions)
        current_recent_actions.append([new_action, current_obs])
        conv.append_message(conv.roles[0], current_obs)

        new_node = ExtendedNode(
            obs=current_obs,
            action=new_action,
            env=self.env,
            state=conv,
            parent=node,
            disaster=disaster,
            recent_actions=current_recent_actions,
            llm_response=agent_response,
            depth=node.depth + 1,
            env_score=new_env_score,
            is_terminal=node.depth + 1 > self.generate_cfg.max_depth or is_terminal
        )
        return new_node

    def expand(self, node):
        if node.is_fully_expanded:
            return
        sampled_nodes = [self._generate(node) for _ in range(self.generate_cfg.n_generate_samples)]
        dedup_nodes, fingerprint = [], set()
        for sample_node in sampled_nodes:
            if sample_node and sample_node.llm_response not in fingerprint:
                fingerprint.add(sample_node.llm_response)
                dedup_nodes.append(sample_node)
        node.children = dedup_nodes
        for child in node.children:
            if child.is_terminal:
                self._backpropagate(child, child.reward)

    def clean(self, s):
        for tok in ['\n', '\t']:
            s = s.replace(tok, ' ')
        return s

    def load(data_path):
        state_dict = mmengine.load(data_path)

        def dict_to_node(data):
            children_data = data.pop('children')
            node = ExtendedNode(**data)
            node.children = [dict_to_node(child) for child in children_data]
            return node

        return dict_to_node(state_dict)
