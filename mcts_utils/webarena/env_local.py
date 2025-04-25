from browser_env import ScriptBrowserEnv
import json


class WebarenaEnvLocal:
    def __init__(self, data_len=200):
        self.env = ScriptBrowserEnv(
            headless=True,  # 设置为 False 可打开浏览器窗口调试
            slow_mo=100,    # 每步等待时间，便于观察
            observation_type="accessibility_tree",
            current_viewport_only=True,
            viewport_size={"width": 1280, "height": 720}
        )
        self.data_len = data_len
        self.config_files = [f"config_files/{i}.json" for i in range(data_len)]
        self.current_idx = None

    def reset(self, idx):
        self.current_idx = idx
        config_file = self.config_files[idx]
        with open(config_file, "r") as f:
            data = json.load(f)
        self.intent = data["intent"]
        self.env.reset(options={"config_file": config_file})
        return config_file

    def observe(self):
        return self.env.observe()

    def step(self, action):
        return self.env.step(action)

    @property
    def conversation_start(self):
        return [{"role": "user", "value": self.intent}]
