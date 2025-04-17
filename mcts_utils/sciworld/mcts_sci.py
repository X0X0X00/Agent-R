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
from mcts_utils.sciworld.eval_utils_sw import findValidActionNew
from mcts_utils.mcts_raw import MCTSNode, MCTSAgent
import os
import re
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


# 新的 MCTS 类，继承原始的 MCTS 类并重写相关功能
class ExtendedMCTS(MCTSAgent):
    def __init__(self, 
                 idx=0,
                 calling=None,
                 encoding=None,
                 max_len=0,
                 model_name=None,
                 logger=None,
                 env=None,
                 generate_cfg=MCTSConfig()):
        super().__init__()
        self.generate_cfg = generate_cfg
        self.calling = calling
        self.max_len = max_len
        self.model_name = model_name
        self.logger = logger
        self.env = env
        self.idx = idx

        

    def search(self, env, conv, recent_actions):
        env_score = 0
        recent_actions_temp = []
        env_reward = 0
        env_done = False
        for action in recent_actions:
            conv.append_message(conv.roles[1], None)
            step_output = self.env.step(action)
            env_state, env_reward, env_done = (
                step_output.state,
                step_output.reward,
                step_output.done,
            )
            current_obs = env.info['observation']
            recent_actions_temp.append(f'({action}, {current_obs})')
            conv.update_last_message(action)
            conv.append_message(conv.roles[0], current_obs)
        init_state = deepcopy(conv)
        init_env = env
        env_score = env_reward
        #
        is_terminal = env_done
        self.root = ExtendedNode(env=init_env, state=init_state, llm_response="ROOT", is_terminal=is_terminal,
                                 recent_actions=recent_actions_temp, env_score=env_score, action="ROOT")
        #
        for iter in range(self.generate_cfg.iterations):
            node = self.root
            if node.is_terminal:
                print(f"Stop at Iter {iter}")
                return
            step_counter = 0
            while node and not node.is_terminal:
                if step_counter >= self.generate_cfg.max_depth:
                    print(f"[⚠️] Exceeded max depth ({self.generate_cfg.max_depth}) at Iter {iter}, breaking...")
                    break
                self.expand(node)
                node = self._select(node)
                step_counter += 1
        return
    
    def _generate(self, node):
        conv = deepcopy(node.state)
        while len(self.calling.encoding.encode(str(conv))) > self.max_len - 60:
            del conv.messages[4:6]
            if conv.messages[4][1].startswith('The preceding task has ended.'):
                del conv.messages[2:4]

        prompt = conv.to_openai_api_messages()
        agent_response = self.calling.llm_func(prompt, self.model_name)
        
        ind = 1
        disaster = False
        agent_response = str(agent_response).strip()
        conv = deepcopy(node.state)
        _ = self.env.reset(self.idx)
        conv.append_message(conv.roles[1], None)
        conv.update_last_message(agent_response.replace(f"Action {ind}:", "Action:").replace(f"Thought {ind}:", "Thought:").strip())

        _ = self.env.reset(self.idx)
        for action in node.recent_actions:
            action_temp = action.split(",")[0].replace("(", "").strip()
            _ = self.env.step(action_temp)

        current_env = self.env
        current_recent_actions = deepcopy(node.recent_actions)


        import re
        if f"Action {ind}:" in agent_response:
            new_action = agent_response.split(f"Action {ind}:")[-1].strip()
            new_action = re.sub(r'^\d+\.\s*', '', new_action)
        elif f"Action:" in agent_response:
            new_action = agent_response.split(f"Action:")[-1].strip()
            new_action = re.sub(r'^\d+\.\s*', '', new_action)
        else:
            new_action = agent_response.split('\n')[-1]

        new_action = findValidActionNew([new_action], current_env, current_env.get_look_around(), current_recent_actions)


        # 立刻加一个判断：
        if new_action in ["-1", "1", "", None] or "No known action matches" in current_env.info["observation"]:
            print(f"[⛔️] Invalid action '{new_action}' → stopping this branch.")
            return ExtendedNode(
                obs="Invalid Action",
                action=new_action,
                env=current_env,
                state=conv,
                parent=node,
                disaster=True,
                recent_actions=current_recent_actions,
                llm_response=agent_response,
                depth=node.depth + 1,
                env_score=0,
                is_terminal=True
            )


        step_output = current_env.step(new_action)
        current_env_state, current_env_reward, current_env_done = (
                step_output.state,
                step_output.reward,
                step_output.done,
            )
        current_obs = current_env_state
        done = current_env_done
        new_env_score = current_env_reward

        if new_env_score < 0:
            is_terminal = True
            new_env_score = 0
            disaster = True
        else:
            is_terminal = done
    
        current_obs = self.clean(current_obs)
        current_recent_actions.append(f'({new_action}, {current_obs})')
        
        print(new_action)
        print(current_obs)
        print(new_env_score)
        conv.append_message(conv.roles[0], current_obs)
        new_node = ExtendedNode(
            obs=current_obs,
            action=new_action,
            env=current_env,
            state=conv,
            parent=node,
            disaster=disaster,
            recent_actions=current_recent_actions,
            llm_response=agent_response.replace(f"Action {ind}:", "Action:").replace(f"Thought {ind}:", "Thought:").strip(),
            depth=node.depth + 1,
            env_score=new_env_score,
            is_terminal=node.depth + 1 > self.generate_cfg.max_depth or is_terminal
        )

        return new_node

    def expand(self, node):
        if not node.is_fully_expanded:
            sampled_nodes = [self._generate(node) for _ in range(self.generate_cfg.n_generate_samples)]
            
            fingerprint, dedup_nodes = set(), []
            for sample_node in sampled_nodes:
                if sample_node.llm_response in fingerprint:
                    continue
                else:
                    fingerprint.add(sample_node.llm_response)
                    dedup_nodes.append(sample_node)
            node.children = dedup_nodes
            for child in node.children:
                if child.is_terminal:
                    self._backpropagate(child, child.reward)

    def clean(self, s):
        clean_toks = ['\n', '\t']
        for tok in clean_toks:
            s = s.replace(tok, ' ')
        return s

    def load(data_path):
        state_dict = mmengine.load(data_path)

        def dict_to_node(data):
            # Recursively construct nodes from dictionary data
            children_data = data.pop('children')
            node = ExtendedNode(**data)
            node.children = [dict_to_node(child) for child in children_data]
            return node

        return dict_to_node(state_dict)