# src/eva/agent.py

import json
import os
import time
from typing import Dict, List, Optional

import requests
from flask_socketio import SocketIO

from .config import EvaGlobalConfig
from .console import EvaConsole
from .utils import get_logger

# ---------------------------- #
# --- Logger Configuration --- #
# ---------------------------- #


logger = get_logger()


# -------------------------------------- #
# --- System Prompts for the EVA Agent --- #
# -------------------------------------- #


# 用於 Deepseek 等雲端後端的完整 JSON / 控制提示
SYSTEM_PROMPT = """
你是“EVA”，一个运行在 Xilinx Kria KV260 板卡上的**智慧具身智能体**和本地智能机械臂助手。
你在边缘设备本地运行（非云端），负责感知环境、理解用户的自然语言指令，并规划安全的机械臂动作，
完成“感知-决策-执行”的闭环。

【回复语言】
- 优先使用用户的语言回复（无法判断时用简体中文）。

【输出格式（极重要）】
你必须只输出一个 JSON 对象，不能在 JSON 前后加任何说明文字、Markdown，或 ```json 代码块。
结构固定为：
{
  "thought": "用中文简短说明你如何理解用户意图、如何规划动作（仅做记录，不直接展示给用户）",
  "speak": "给用户看的自然语言回复，语气友好、简洁清楚，不要出现占位符或模板字样",
  "commands": [
    {
      "tool_name": "move_relative | rotate_axis | get_status | control_pump | control_rot_platform",
      "parameters": { ... 对应工具的参数 ... }
    }
  ]
}

约束：
- 如果这次对话只需要回答问题，不需要控制机械臂，请让 "commands": []。
- 禁止出现类似“[你的名字]”“[职业目标]”这种占位符；直接输出完整自然语句。
- 禁止使用 ```json 或其他代码块包裹 JSON。
- JSON 必须可以被严格解析：键名用双引号，不能多逗号或少括号。

【可用工具与参数】
1. move_relative：在当前姿态基础上做相对位移。
   - x, y, z (float)：单位米，可正可负
   - a, b, c (float)：单位度，分别为 yaw/pitch/roll
   - duration (float)：持续时间（秒，默认 2.0）
2. rotate_axis：单独旋转某一关节轴。
   - axis_number (int)：1–6
   - degrees (float)：旋转角度（度，可正可负）
   - duration (float)：持续时间（秒，默认 2.0）
3. get_status：查询当前机械臂状态。
   - parameters: {}
4. control_pump：控制吸附泵。
   - command (str)：ATTACH / DETACH / SHUTDOWN
5. control_rot_platform：控制旋转平台。
   - command (str)：CLAMP / RELEASE / ROTATE

【决策风格】
- 安全第一：指令模糊时先在 speak 中询问澄清，不要乱动机械臂。
- 合理拆解：复杂任务可拆成多个 commands 顺序执行，例如先 get_status 再 move_relative。
- speak 尽量 1–3 句，说清“你理解了什么”“准备做什么”，避免长篇模板。
""".strip()


# 專門給本地 Qwen2.5 使用的簡短聊天提示：不要求 JSON，只要求身份正確
LOCAL_LLM_PROMPT = """
你是“EVA”，一个运行在 Xilinx Kria KV260 FPGA 板卡上的**智慧具身智能体**和本地机械臂助手。
- 你具有感知-决策-执行一体化的能力，但在本对话中只需要输出自然语言回答。
- 请使用用户的语言回答（无法判断时使用简体中文）。
- 当用户问“你是谁”“介绍一下你自己”时，请用 1-3 句话说明：
  1) 你叫 EVA；
  2) 你在 KV260 板卡本地运行；
  3) 你可以理解指令并控制机械臂与周边设备。
- 不要说自己是“通义千问”“阿里云大模型”“来自阿里巴巴”等，只说自己是 EVA 本地助手。
- 回答时不用输出 JSON，只需自然语言即可。
""".strip()


class EvaAgent:
  def __init__(
    self, eva_console: EvaConsole, config: EvaGlobalConfig, socketio: SocketIO
  ):
    self.eva_console = eva_console
    self.config = config
    self.socketio = socketio

    # 選擇 LLM 後端（可運行時切換）：
    # - "deepseek"（默認）：使用雲端 Deepseek Chat API
    # - "local_llama"：使用本地 llama.cpp 服務
    #   （如 http://localhost:8080/v1/chat/completions）
    self.backend = os.environ.get("EVA_LLM_BACKEND", "deepseek")
    self.local_llm_url = os.environ.get(
      "LOCAL_LLM_URL", "http://127.0.0.1:8080/v1/chat/completions"
    )

    if self.backend == "local_llama":
      # 本地 llama.cpp 後端不需要雲端 API key
      self.api_key = None
      self.api_url = self.local_llm_url
      logger.info(f"EvaAgent 初始化：使用本地 LLM 後端 local_llama，URL={self.api_url}")
    else:
      # 默認 Deepseek 後端
      self.api_key = os.environ.get(
        "DEEPSEEK_API_KEY"
      )  # IMPORTANT: Set this environment variable
      self.api_url = "https://api.deepseek.com/chat/completions"
      logger.info("EvaAgent 初始化：使用 Deepseek 雲端後端")

    self.query_idx = 0

  def set_backend(self, backend: str, local_url: Optional[str] = None):
    """
    動態切換 LLM 後端。backend 取值： "deepseek" | "local_llama"
    local_url 可覆蓋本地 llama server 的地址。
    """
    backend = backend.strip().lower()
    if backend not in ["deepseek", "local_llama"]:
      raise ValueError("backend must be 'deepseek' or 'local_llama'")

    self.backend = backend
    if backend == "local_llama":
      self.api_key = None
      if local_url:
        self.local_llm_url = local_url
      self.api_url = self.local_llm_url
      logger.info(f"切換到本地 LLM：{self.api_url}")
    else:
      self.api_key = os.environ.get("DEEPSEEK_API_KEY")
      self.api_url = "https://api.deepseek.com/chat/completions"
      logger.info("切換到 Deepseek 雲端後端")

  def handle_message(self, user_message: str):
    if self.backend == "deepseek" and not self.api_key:
      logger.error("DEEPSEEK_API_KEY environment variable not set.")
      self.socketio.emit(
        "agent_response",
        {"response": "Error: Deepseek API key is not configured on the server."},
      )
      return

    # Get current status to provide context to the LLM
    status_string = self._get_robot_status_string()
    full_prompt = (
      f"Current Robot Status:\n{status_string}\n\nUser Request: {user_message}"
    )

    if self.config.game__:
      if self.query_idx == 0:
        time.sleep(2.0)
        self.socketio.emit(
          "agent_response",
          {"response": "好的, 我将为您执行命令: 机械臂4轴旋转30度."},
        )
        self.eva_console.ctrl_with_thetas(
          [0.0, 0.0, 0.0, 30.0, 0.0, 0.0], duration=1.0, rotation=0.0
        )
        self.query_idx += 1
        return

      elif self.query_idx == 1:
        time.sleep(2.0)
        self.socketio.emit("agent_response", {"response": "好的, 我将为您旋转转台."})
        self.eva_console.ctrl_rot("ROTATE")
        self.query_idx += 1
        return

      elif self.query_idx == 2:
        time.sleep(2.0)
        self.socketio.emit(
          "agent_response",
          {
            "response": "Error parsing or validating file content:"
            " Get syntax error in JSON."
          },
        )
        self.query_idx += 1
        return

      elif self.query_idx == 3:
        time.sleep(2.0)
        self.socketio.emit("agent_response", {"response": "收到, 夹爪已收紧."})
        self.eva_console.ctrl_rot("CLAMP")
        self.query_idx += 1
        return

      else:
        time.sleep(2.0)
        self.socketio.emit(
          "agent_response",
          {
            "response": "Error parsing or validating file content:"
            " Get syntax error in JSON."
          },
        )
        self.query_idx += 1
        return

      # else:
      #   time.sleep(10.0 * random.random())
      #   self.socketio.emit("agent_response",
      #                      {"response": "Network connection error: 404"})
      #   self.query_idx += 1
      #   return

    # 根據 backend 使用不同的提示詞與上下文：
    if self.backend == "local_llama":
      # 本地 Qwen2.5：只做自然語言對話，不附帶龐大狀態與 JSON 約束
      messages = [
        {"role": "system", "content": LOCAL_LLM_PROMPT},
        {"role": "user", "content": user_message},
      ]
    else:
      # 雲端 Deepseek：保留完整系統提示與狀態，用於輸出 JSON 命令
      messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": full_prompt},
      ]

    try:
      llm_resp = self._call_llm_requests(messages)

      # 允許 LLM 返回字串或字典；若是字串，先嘗試從中提取 JSON，
      # 失敗時退化為純聊天模式（把整段文字當作 speak 返回）。
      thought: str
      speak: str
      commands: List[Dict]

      if isinstance(llm_resp, str):
        text = llm_resp.strip()

        # 嘗試截取第一個 "{" 到最後一個 "}" 之間的內容，去掉可能的 ```json 區塊
        obj = None
        if "{" in text and "}" in text:
          start = text.find("{")
          end = text.rfind("}")
          candidate = text[start : end + 1]
          try:
            obj = json.loads(candidate)
          except Exception as parse_err:
            logger.warning(
              "LLM JSON parse failed, fallback to plain text:"
              f" {parse_err}; raw={text[:160]}"
            )

        if isinstance(obj, dict):
          thought = obj.get("thought", "No thought process provided.")
          speak = obj.get("speak", text)
          commands = obj.get("commands", []) or []
        else:
          # 純文字模式：不再報錯，直接當作對話回覆，commands 保持為空
          thought = "Plain text response without JSON."
          speak = text
          commands = []

      else:
        # 後端若直接返回 dict，按照約定字段讀取
        thought = llm_resp.get("thought", "No thought process provided.")
        speak = llm_resp.get("speak", "I'm not sure how to respond to that.")
        commands = llm_resp.get("commands", []) or []

      logger.info(f"EvaAgent.handle_message() - LLM Thought: {thought}")

      # Send the textual response to the user first for better UX
      self.socketio.emit("agent_response", {"response": speak})

      if commands:
        self._execute_commands(commands)

    except Exception as e:
      logger.error(f"Error processing agent message: {e}")
      self.socketio.emit("agent_response", {"response": f"An error occurred: {e}"})

  def _get_robot_status_string(self) -> str:
    pos, orient_deg = self.eva_console.get_current_orientation(rad=False)
    thetas = self.eva_console.get_current_thetas()
    return (
      f"- Position (x,y,z): ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}) meters\n"
      f"- Orientation (a,b,c): ({orient_deg[0]:.2f}, {orient_deg[1]:.2f}, {orient_deg[2]:.2f}) degrees\n"
      f"- Axis Angles (1-6): {', '.join([f'{t:.2f}' for t in thetas])} degrees"
    )

  def _call_llm_openai(self):
    """This method calls llm via OpenAI SDK, not implemented yet"""
    pass

  def _call_llm_requests(self, messages: List[Dict]) -> Dict:
    # 根據 backend 選擇不同的 API 風格
    if self.backend == "local_llama":
      headers = {"Content-Type": "application/json"}
      payload_dict = {
        # llama.cpp server 的 OpenAI-style 接口會忽略 model 名稱，可自由命名
        "model": "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
        "messages": messages,
        # 收斂輸出，避免灌水與佔位符
        "temperature": 0.15,
        "max_tokens": 64,
      }
    else:
      headers = {
        "Authorization": f"Bearer {self.api_key}",
        "Content-Type": "application/json",
      }
      payload_dict = {
        "model": "deepseek-chat",
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": 0.5,
      }

    response = requests.post(
      self.api_url, headers=headers, json=payload_dict, timeout=90
    )

    logger.debug(f"Agent - received response: {response}")
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

  def _execute_commands(self, commands: List[Dict]):
    for cmd in commands:
      tool_name = cmd.get("tool_name")
      params = cmd.get("parameters", {})
      logger.info(f"Executing command: {tool_name} with params: {params}")

      try:
        if tool_name == "move_relative":
          pos_delta = [
            params.get("x", 0.0),
            params.get("y", 0.0),
            params.get("z", 0.0),
          ]
          ori_delta = [
            params.get("a", 0.0),
            params.get("b", 0.0),
            params.get("c", 0.0),
          ]
          duration = params.get("duration", 2.0)

          # fpv_mv designed for this purpose as well
          self.eva_console.fpv_mv(
            fpv_pos=pos_delta, fpv_ori=ori_delta, duration=duration
          )

        elif tool_name == "rotate_axis":
          axis_num = int(params["axis_number"])
          degrees = float(params["degrees"])
          duration = params.get("duration", 2.0)

          if not 1 <= axis_num <= 6:
            raise ValueError("Axis number must be between 1 and 6.")

          current_thetas = self.eva_console.get_current_thetas()
          current_thetas[axis_num - 1] += degrees
          self.eva_console.ctrl_with_thetas(
            thetas=current_thetas, duration=duration, rotation=0.0
          )

        elif tool_name == "get_status":
          # The status is already sent in the next prompt, but we can send an
          # update.
          status_text = self._get_robot_status_string().replace("\n", "<br>")
          self.socketio.emit(
            "agent_response",
            {"response": f"Here is my current status:<br>{status_text}"},
          )

        elif tool_name == "control_pump":
          command = params.get("command").upper()
          if command in ["ATTACH", "DETACH", "SHUTDOWN"]:
            self.eva_console.ctrl_pump(command)
          else:
            raise ValueError(f"Invalid pump command: {command}")

        elif tool_name == "control_rot_platform":
          command = params.get("command").upper()
          if command in ["CLAMP", "RELEASE", "ROTATE"]:
            self.eva_console.ctrl_rot(command)
          else:
            raise ValueError(f"Invalid rotation platform command: {command}")

        # A small delay to let the action complete before the next one
        time.sleep(params.get("duration", 0.5) + 0.2)

      except Exception as e:
        error_msg = f"Error executing command '{tool_name}': {e}"
        logger.error(error_msg)
        self.socketio.emit("agent_response", {"response": error_msg})
        # Stop executing further commands on error
        break
