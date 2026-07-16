import ast
import logging
import os
import re
from base64 import b64encode
from PIL import Image
from io import BytesIO
from typing import Dict, List

from .prompt.accessibility_tree_handle import linearize_accessibility_tree, trim_accessibility_tree
from .prompt.grounding_agent import GroundingAgent as Agent
from .tools.package.google_chrome import BrowserTools
from .prompt.procedural_memory import Prompt
from ..stall_recovery import (
    HintBank,
    StallDetector,
    _executed_action_key,
    _strip_mangle_noise,
    call_dumps,
    format_stall_intervention,
    pseudo_action,
    screenshot_thumbnail,
    thumb_similarity,
)

logger = logging.getLogger("desktopenv.agent")

pure_text_settings = ["a11y_tree"]

def _recovery_fingerprint(stage: str, exc) -> str:
    """Discriminative, normalized recovery fingerprint: stage + exception class +
    message (quotes/numbers masked). Ensures three DIFFERENT interface failures are
    NOT collapsed into one L2 same-template count."""
    msg = getattr(exc, "msg", None) or str(exc)
    msg = re.sub(r"'[^']*'", "'X'", msg)
    msg = re.sub(r'"[^"]*"', '"X"', msg)
    msg = re.sub(r"\b\d+\b", "N", msg)
    return "{}:{}: {}".format(stage, type(exc).__name__, re.sub(r"\s+", " ", msg).strip())[:120]

def resize_image(image, w, h):
    img = Image.open(BytesIO(image))
    # resize to max_pixel_num max_pixels
    img = img.resize((w, h))
    buf = BytesIO()
    img.save(buf, format='PNG')
    img_bytes = buf.getvalue()
    return img_bytes

def parse_code_from_string(input_string):
    # input_string = "\n".join([line.strip() for line in input_string.split(';') if line.strip()])
    if input_string.strip() in ["WAIT", "DONE", "FAIL"]:
        return [input_string.strip()]

    # This regular expression will match both ```code``` and ```python code```
    # and capture the `code` part. It uses a non-greedy match for the content inside.
    pattern = r"```(?:\w+\s+)?(.*?)```"
    # Find all non-overlapping matches in the string
    matches = re.findall(pattern, input_string, re.DOTALL)

    # The regex above captures the content inside the triple backticks.
    # The `re.DOTALL` flag allows the dot `.` to match newline characters as well,
    # so the code inside backticks can span multiple lines.

    # matches now contains all the captured code snippets

    codes = []

    for match in matches:
        match = match.strip()
        commands = ["WAIT", "DONE", "FAIL"]  # fixme: updates this part when we have more commands

        if match in commands:
            codes.append(match.strip())
        elif match.split("\n")[-1] in commands:
            if len(match.split("\n")) > 1:
                codes.append("\n".join(match.split("\n")[:-1]))
            codes.append(match.split("\n")[-1])
        else:
            codes.append(match)

    return codes


class AutoGLMAgent:
    def __init__(
        self,
        action_space="autoglm_computer_use",
        observation_type="a11y_tree",
        max_trajectory_length=3,
        a11y_tree_max_items=300,
        with_image: bool = True,
        screen_size = (1920, 1080),
        image_size=(1920, 1080),
        with_atree: bool = False,
        glm41v_format: bool = True,
        relative_coordinate: bool = True,
        client_password="password",
        gen_func=None,
        tool_in_sys_msg: bool = True,
        omni_data_dir=None,
        omni_llm_model="autoglm-os",
        omni_top_k: int = 5,
        error_ledger=None,
        use_recovery: bool = False,
        stall_recovery: str = "off",
        stall_hint_bank: str = None,
    ):
        self.action_space = action_space
        self.observation_type = observation_type
        assert action_space in ["autoglm_computer_use"], "Invalid action space"
        assert observation_type in ["a11y_tree"], "Invalid observation type"
        self.max_trajectory_length = max_trajectory_length
        self.a11y_tree_max_items = a11y_tree_max_items
        self.with_image = with_image
        self.screen_size = screen_size
        self.image_size = image_size
        self.with_atree = with_atree
        self.glm41v_format = glm41v_format
        self.relative_coordinate = relative_coordinate
        self.client_password = client_password
        self.gen_func = gen_func
        self.tool_in_sys_msg = tool_in_sys_msg

        self._omni_data_dir = omni_data_dir
        self._omni_llm_model = omni_llm_model
        self._omni_top_k = omni_top_k
        self._init_omni()
        self.error_ledger = error_ledger

        self.tool_list = {
            "libreoffice_calc": "CalcTools",
            "libreoffice_impress": "ImpressTools",
            "libreoffice_writer": "WriterTools",
            "code": "CodeTools",
            "vlc": "VLCTools",
            "google_chrome": "BrowserTools",
        }
        
        Agent.relative_coordinate = relative_coordinate

        # Intra-task action-interface repair -- independent of Omni / ledger / parser.
        self.use_recovery = use_recovery

        # Stall-triggered retrieval-augmented recovery -- independent of use_recovery
        # (the frozen interface-repair arm) and of the ledger.
        # off | replan | hint | forbid. With "off" the prompt stays byte-identical
        # to baseline.
        self.stall_recovery = stall_recovery if stall_recovery in ("off", "replan", "hint", "forbid") else "off"
        self._stall_detector = StallDetector() if self.stall_recovery != "off" else None
        # Forbid arm: the exact repeated action is blocked only while the screen
        # remains equivalent to the state where the stall was detected.
        self._stall_forbidden = None
        self._stall_bank = None
        if self.stall_recovery == "hint" and stall_hint_bank:
            try:
                self._stall_bank = HintBank(stall_hint_bank)
            except Exception as e:
                logger.warning("StallRecovery: hint bank unavailable (%s); running as replan arm", e)

        self.contents = []

    @property
    def turn_number(self):
        return len(self.contents)

    def prepare(self, instruction: str, obs: Dict, history: List, last_result: str = "", repair: bool = False, stall_context: str = "") -> List:
        """
        Predict the next action(s) based on the current observation.
        """
        if "exe_result" in obs and not last_result:
            last_result = obs["exe_result"]
            if self.contents:
                self.contents[-1]["exe_result"] = last_result

        cur_app = obs["cur_app"]
        logger.info(f"current app is {cur_app}")

        if cur_app:
            tool_name = cur_app.strip().lower().replace("-", "_")
            tool_name = tool_name if tool_name in self.tool_list.keys() else None
        else:
            tool_name = None

        setup_prompt, func_def_prompt, note_prompt = Prompt.construct_procedural_memory(
            Agent, app_name=tool_name, client_password=self.client_password, with_image=self.with_image, with_atree=self.with_atree, relative_coordinate=self.relative_coordinate, glm41v_format=self.glm41v_format
        )
        if self.tool_in_sys_msg:
            system_message = setup_prompt + "\n\n" + func_def_prompt + "\n\n" + note_prompt
        else:
            system_message = setup_prompt + "\n\n" + note_prompt
        system_message += "\n\n**IMPORTANT** You are asked to complete the following task: {}".format(instruction)

        # Cross-task memory injection. The backend decides *whether* to surface
        # anything (v2 gates on the live context below); we only route the result.
        mem_context, mem_target = "", "system"
        if self.error_ledger is not None and tool_name is not None:
            recent_actions = [str(c.get("action", "")) for c in self.contents[-3:]]
            mem_context = self.error_ledger.retrieve(
                tool_name,
                instruction=instruction,
                last_result=last_result,
                recent_actions=recent_actions,
                # +1 so the logged step_idx matches lib_run_single's record (step_idx+1),
                # i.e. the injection on this turn aligns with the transition it precedes.
                step_idx=self.turn_number + 1,
                obs=obs,
            )
            mem_target = getattr(self.error_ledger, "inject_target", "system")
        if mem_context and mem_target == "system":
            system_message += "\n\n" + mem_context

        messages = [
            {
                "role": "system",
                "content": system_message,
            }
        ]
        messages.extend(history)

        if obs["apps"]:
            app_str = "Window ID    App Name    Title\n"
            for window_id, app in obs["apps"].items():
                app_str += f"{window_id}    {app['app_name']}    {app['title']}\n"
        else:
            app_str = "None"

        last_result = last_result.strip() if last_result else "None"
        last_result = last_result[:2000] + "..." if len(last_result) > 2000 else last_result

        tree = linearize_accessibility_tree(obs["accessibility_tree"], "Ubuntu")
        tree = trim_accessibility_tree(tree, 300)

        app_info = obs["app_info"].strip() if obs["app_info"] else "None"
        app_info = app_info[:5000] + "..." if len(app_info) > 5000 else app_info

        prompt = "* Apps: {}\n\n* Current App: {}{}\n\n* App Info: {}\n\n* Previous Action Result: {}".format(
            app_str.strip(),
            obs["cur_window_id"].strip() if obs["cur_window_id"] in app_str else "None",
            '\n\n* A11y Tree: {}'.format(tree.strip()) if self.with_atree else "",
            app_info,
            last_result if last_result else "None",
        ) + (
            "\n\n" + func_def_prompt if not self.tool_in_sys_msg else ""
        )

        # state-dependent memo (v2) rides in the user turn, not the system prompt
        if mem_context and mem_target == "user":
            prompt += "\n\n" + mem_context

        # stall-recovery intervention also rides in the user turn (never the
        # system prompt, never the repair contract below).
        if stall_context:
            prompt += "\n\n" + stall_context

        # action-interface repair (L2): on a same-step regeneration, append a hard
        # output contract. The caller passes only a short recent history window so the
        # model keeps local task state without replaying the full echo-prone trace.
        if repair:
            prompt += (
                "\n\n* Recovery: your previous response did NOT enter the execution channel "
                "(it produced no parseable/executable action). Keep your current intent and "
                "return ONLY one fenced python code block containing a single executable "
                "action for the current screen. No explanation, no prose, do not repeat text."
            )

        content = [{"type": "text", "text": prompt}]
        if self.with_image and obs.get('screenshot'):
            screenshot = resize_image(obs['screenshot'], self.image_size[0], self.image_size[1])
            content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64encode(screenshot).decode('utf-8')}",
                        "detail": "high",
                    },
                }
            ] + content

        messages.append({"role": "user", "content": content})

        return messages

    def execute(self, response, obs):
        # ComputerRL adapter for the recovery monitor: did the action cross the
        # parser/dispatch interface? parser empty -> no; grounded command fails to
        # compile -> no; otherwise yes. (Tool runtime errors are NOT interface
        # failures -- they crossed the interface and got real feedback.)
        interface_ok, parse_exc = True, None
        try:
            actions = parse_code_from_string(response)
            action = actions[0]
            logger.info(f"The pesudo action is {action}")

            actions = self._ground_action(action, obs)
        except Exception as e:
            print("Failed to parse action from response", e)
            actions = []
            interface_ok, parse_exc = False, e

        # Interface probe (only when recovery is on -- baseline does zero extra work).
        # Returns whether the action crossed the interface + a normalized failure
        # fingerprint; the streak update / repair orchestration happen in predict().
        fingerprint = ""
        if self.use_recovery:
            if parse_exc is not None:
                # discriminative fingerprint (stage+class+msg); distinct failures keep
                # distinct fingerprints (for the repair log / failure attribution).
                fingerprint = _recovery_fingerprint("parse", parse_exc)
            elif actions and isinstance(actions[0], str):
                # any STRING command must compile to have crossed the interface; a dict/
                # object action (eval'd Agent.*/BrowserTools.*) is already a valid action.
                interface_ok, fingerprint = self._interface_check(actions[0])

        return actions, interface_ok, fingerprint

    def _ground_action(self, action, obs):
        action = action.strip()
        if action.startswith("Agent."):
            return [eval(action)]
        if action.startswith("BrowserTools."):  # TODO: special check for BrowserTools
            return [eval(action)]
        actions = Agent.tool_commands(action, obs["cur_app"].strip().replace("-", "_").lower())
        logger.info(f"The grounded action is {actions[0]}")
        return actions

    @staticmethod
    def _interface_check(command):
        """compile() the grounded command without running it. ANY compile rejection
        (SyntaxError / IndentationError / TabError / ValueError ...) means the dispatched
        command is not valid code => the action never crossed the execution interface
        (our serialization broke), NOT tool feedback."""
        try:
            compile(command, "<recovery-interface-check>", "exec")
            return True, ""
        except Exception as e:
            return False, _recovery_fingerprint("compile", e)

    # ---- action-interface repair --------------------------------------------------
    MAX_REPAIR = 2   # L2 model regenerations per step; L1 is zero-LLM contract repair.

    def _gen(self, messages):
        assert self.gen_func is not None, "gen_func is not set"
        for _ in range(3):
            try:
                return self.gen_func(messages)
            except Exception as e:
                logger.error("Failed to call gen_func, Error: " + str(e))
        raise RuntimeError("Failed to call gen_func after retries")

    @staticmethod
    def _content_text(obj):
        if isinstance(obj, str):
            return obj
        if isinstance(obj, list):
            return "\n".join(AutoGLMAgent._content_text(x) for x in obj)
        if isinstance(obj, dict):
            if "text" in obj:
                return AutoGLMAgent._content_text(obj["text"])
            if "content" in obj:
                return AutoGLMAgent._content_text(obj["content"])
        return ""

    @classmethod
    def _response_text(cls, response):
        if isinstance(response, (list, dict)):
            return cls._content_text(response)
        text = str(response)
        stripped = text.strip()
        if stripped.startswith(("[", "{")):
            try:
                parsed = ast.literal_eval(stripped)
            except Exception:
                return text
            parsed_text = cls._content_text(parsed)
            return parsed_text or text
        return text

    def _allowed_action_prefixes(self, obs):
        prefixes = ["Agent"]
        tool_name = (obs.get("cur_app") or "").strip().lower().replace("-", "_")
        if tool_name in self.tool_list:
            prefixes.append(self.tool_list[tool_name])
        return prefixes

    @staticmethod
    def _strip_code_noise(code):
        code = (code or "").strip()
        code = re.sub(r"^\s*python\\n", "", code, flags=re.IGNORECASE).strip()
        code = re.sub(r"^\s*python\s*\n", "", code, flags=re.IGNORECASE).strip()
        if code.endswith("\\n"):
            code = code[:-2].strip()
        return code

    @staticmethod
    def _find_action_calls(text, prefixes):
        if not text:
            return []
        prefix_re = "|".join(re.escape(p) for p in prefixes)
        # Narrow by construction: one current-response call with no nested parentheses.
        call_re = re.compile(
            rf"\b(?:{prefix_re})\.\w+\((?:[^()]|'[^'\\]*(?:\\.[^'\\]*)*'|\"[^\"\\]*(?:\\.[^\"\\]*)*\")*\)",
            re.DOTALL,
        )
        return [m.group(0).strip() for m in call_re.finditer(text)]

    def _contract_repair(self, response, obs):
        """L1: keep the model's current action, repair only the submission contract."""
        text = self._response_text(response)
        prefixes = self._allowed_action_prefixes(obs)

        candidates = []
        for code in parse_code_from_string(text):
            code = self._strip_code_noise(code)
            candidates.extend(self._find_action_calls(code, prefixes))
        if not candidates:
            candidates = self._find_action_calls(text, prefixes)

        unique = list(dict.fromkeys(candidates))
        if len(unique) != 1:
            logger.info("ActionRepair(L1): skip contract repair; candidates=%d", len(unique))
            return None

        action = unique[0]
        try:
            actions = self._ground_action(action, obs)
            if actions and isinstance(actions[0], str):
                interface_ok, _fingerprint = self._interface_check(actions[0])
                if not interface_ok:
                    logger.info("ActionRepair(L1): candidate failed interface check: %s", action)
                    return None
        except Exception as e:
            logger.info("ActionRepair(L1): candidate failed grounding: %s", e)
            return None

        repaired_response = "```python\n{}\n```".format(action)
        logger.info("ActionRepair(L1): contract repaired action=%s", action)
        return repaired_response, actions

    def format_history(self, current_exe_result="", max_turns=30):
        history = []
        # for ix in range(self.turn_number):
        #     if ix == 0:
        #         env_input = "**Environment State (Omitted)**"
        #     else:
        #         env_input = (
        #             f"**Environment State (Omitted)**\nPrevious Action Result: {self.contents[ix - 1]['exe_result']}"
        #         )

        #     env_input = env_input[:2000] + "..." if len(env_input) > 2000 else env_input
        #     response = (
        #         self.contents[ix]["response"][:1500] + "..."
        #         if len(self.contents[ix]["response"]) > 1500
        #         else self.contents[ix]["response"]
        #     )
        #     history.append({"role": "user", "content": [{"type": "text", "text": env_input}]})
        #     history.append({"role": "assistant", "content": [{"type": "text", "text": response}]})

        # revise for the omni memory implenmentation

        if self._omni is not None:
            query_a = self.contents[-1]["instruction"] if self.contents else ""
            query_b = current_exe_result
            indices = self._omni_retrieve(query_a, query_b)
        else:
            indices = range(self.turn_number)  # fallback to sliding window if omni is not available
    
        for ix in indices:  
            if ix == 0:
                env_input = "**Environment State (Omitted)**"
            else:
                env_input = (
                    f"**Environment State (Omitted)**\nPrevious Action Result: {self.contents[ix - 1]['exe_result']}"
                )

            env_input = env_input[:2000] + "..." if len(env_input) > 2000 else env_input
            response = (
                self.contents[ix]["response"][:1500] + "..."
                if len(self.contents[ix]["response"]) > 1500
                else self.contents[ix]["response"]
            )
            history.append({"role": "user", "content": [{"type": "text", "text": env_input}]})
            history.append({"role": "assistant", "content": [{"type": "text", "text": response}]})

        return history[-max_turns * 2:]
    
    def _omni_retrieve(self, query_a, query_b):
        indices = []
        for query in filter(None, [query_a, query_b]):
            try:
                result = self._omni.query(query, top_k=self._omni_top_k, auto_expand=False)
            except Exception as e:
                logger.warning(f"Omni query failed: {e}")
                continue
            for item in (result.items if hasattr(result, "items") else []):
                logger.info(f"Omni item score={item.get('score', 0):.3f} tags={item.get('tags')}")
                for tag in (item.get("tags") or []):
                    if isinstance(tag, str) and tag.startswith("step:"):
                        try:
                            indices.append(int(tag.split(":")[1]))
                            break
                        except (ValueError, IndexError):
                            pass
        if not indices:
            logger.info(f"Omni retrieve empty, fallback to sliding window (turn={self.turn_number})")
            return list(range(self.turn_number))[-self._omni_top_k:]
        result_indices = sorted(set(i for i in indices if 0 <= i < self.turn_number))
        logger.info(f"Omni retrieved indices={result_indices} from {self.turn_number} turns")
        return result_indices

    def _refresh_stall_forbidden(self, obs: Dict) -> None:
        """Keep the exact-action ban local to the screen that produced it."""
        if self._stall_forbidden is None:
            return
        current_thumb = screenshot_thumbnail(obs.get("screenshot"))
        similarity = thumb_similarity(
            self._stall_forbidden.get("state_thumb"), current_thumb
        )
        threshold = self._stall_detector.state_threshold
        if similarity is None or similarity < threshold:
            logger.info("StallContractCleared: %s", {
                "reason": "missing_state" if similarity is None else "state_changed",
                "state_similarity": None if similarity is None else round(similarity, 6),
            })
            self._stall_forbidden = None

    def _stall_update(self, obs: Dict) -> str:
        """Feed the stall detector with the PREVIOUS action's outcome (the current
        obs carries its exe_result and the post-action screenshot) and return the
        intervention block for THIS turn when a stall fires. Best-effort like the
        repair arm: any failure degrades to no intervention, never to a crash."""
        try:
            if not self.contents:
                return ""
            prev = self.contents[-1]
            signal = self._stall_detector.update(
                executed_action=prev.get("action"),
                exe_result=str(obs.get("exe_result", "") or ""),
                screenshot=obs.get("screenshot"),
            )
            if signal is None:
                return ""
            # The hint rides only on the FIRST fire of a task: if the agent is
            # still stuck after a hinted replan, later fires drop the hint so a
            # bad reference cannot keep anchoring the re-plan (v31 lesson).
            hint = None
            if self._stall_bank is not None and signal["fire_index"] == 1:
                recent = [str(c.get("response", "")) for c in self.contents[-4:]]
                hint = self._stall_bank.retrieve(
                    app=obs.get("cur_app") or "",
                    instruction=str(prev.get("instruction", "")),
                    recent_responses=recent,
                    screenshot=obs.get("screenshot"),
                )
            logger.info("StallRecovery: %s", {
                "rule": signal["rule"],
                "fire_index": signal["fire_index"],
                "step_index": signal["step_index"],
                "state_similarity": round(signal["state_similarity"], 6),
                "mode": self.stall_recovery,
                "stalled": signal.get("stalled_action", ""),
                "hint": None if hint is None else {
                    "donor_task": hint.get("task_id"),
                    "donor_domain": hint.get("domain"),
                    "score": hint.get("score"),
                },
            })
            forbid = self.stall_recovery == "forbid"
            # The contract must name the action in the MODEL's own submitted form
            # (e.g. Agent.click([100, 200])), not the grounded pyautogui command the
            # detector keys on -- the model has never seen the grounded string.
            submitted = _strip_mangle_noise(pseudo_action(str(prev.get("response", ""))))
            if submitted:
                signal["stalled_action"] = " ".join(submitted.split())[:120]
            text = format_stall_intervention(signal, hint, forbid=forbid)
            if forbid and signal.get("stalled_key"):
                self._stall_forbidden = {
                    "key": signal["stalled_key"],
                    "action": signal.get("stalled_action", ""),
                    "context": text,
                    "state_thumb": screenshot_thumbnail(obs.get("screenshot")),
                }
            return text
        except Exception as e:
            logger.warning("StallRecovery aborted (%s); continuing without intervention", e)
            return ""

    def predict(self, instruction: str, obs: Dict) -> List:
        if self._stall_detector is not None:
            self._refresh_stall_forbidden(obs)
            active_context = (
                self._stall_forbidden.get("context", "")
                if self._stall_forbidden is not None else ""
            )
            fresh_context = self._stall_update(obs)
            stall_context = fresh_context or active_context
        else:
            stall_context = ""
        history = self.format_history(obs.get("exe_result", ""))
        messages = self.prepare(instruction, obs, history, stall_context=stall_context)

        response = self._gen(messages)
        logger.info("RESPONSE: %s", response)
        actions, interface_ok, fingerprint = self.execute(response, obs)

        # Action-interface repair is BEST-EFFORT: it may only help, never crash a task the
        # baseline handled. The whole L1/L2 block is guarded -- on ANY failure we restore
        # the un-repaired (baseline) action, so use_recovery can never regress robustness.
        base_response = response
        base_actions, base_ok, base_fp = actions, interface_ok, fingerprint
        repair_level = None
        attempt = 0
        if self.use_recovery and not interface_ok:
            try:
                repaired = self._contract_repair(response, obs)   # L1: zero-LLM contract repair
                if repaired is not None:
                    # Execute the repaired action but keep the model's original response
                    # in contents/history/logs: overwriting it with the bare code block
                    # strips the <think> trail and induces replay loops on later turns.
                    _, actions = repaired
                    interface_ok, fingerprint = True, ""
                    repair_level = "L1_contract"

                # L2: if L1 cannot isolate a single current-response action, regenerate once
                # with a short recent-history window (keeps local task state) instead of the
                # full echo-prone trace or a totally blank history.
                while not interface_ok and attempt < self.MAX_REPAIR:
                    repair_history = history[-2:] if history else []
                    repair_msgs = self.prepare(instruction, obs, history=repair_history, repair=True)
                    logger.info("ActionRepair(L2): attempt=%d/%d", attempt + 1, self.MAX_REPAIR)
                    response = self._gen(repair_msgs)
                    logger.info("RESPONSE(repair-L2): %s", response)
                    actions, interface_ok, fingerprint = self.execute(response, obs)
                    repair_level = "L2_regen"
                    attempt += 1
            except Exception as e:
                logger.warning("ActionRepair aborted (%s); falling back to baseline action", e)
                # restore the response together with the action: a first L2 pass may
                # already have overwritten it, and history/logs must never carry a
                # response whose action was not the one executed.
                response = base_response
                actions, interface_ok, fingerprint = base_actions, base_ok, base_fp
                repair_level = "aborted"

        # Per-step diagnostic for failure attribution. No early stop: if repair didn't
        # fix it, fall through like a normal parse failure -- we only ever make the action
        # CORRECT, never force a FAIL.
        if self.use_recovery:
            logger.info("IntraRecovery: %s", {"interface_ok": interface_ok,
                                              "fingerprint": fingerprint,
                                              "repaired": repair_level is not None,
                                              "repair_level": repair_level,
                                              "attempts": attempt})

        # State-local stall contract (forbid arm only). The same model proposes one
        # action; code rejects the exact stalled action and permits one regeneration.
        # If no different executable candidate is produced, dispatch nothing rather
        # than knowingly execute the blocked action again.
        if self._stall_forbidden is not None:
            forbidden = self._stall_forbidden
            violated = False
            try:
                forbidden_dumps = call_dumps(forbidden.get("action", ""))

                def _violates(resp_obj, acts):
                    # Check both the grounded command and exact submitted statements:
                    # a response cannot evade the ban by appending another call.
                    if any(
                        _executed_action_key(action) == forbidden["key"]
                        for action in (acts or [])
                    ):
                        return True
                    submitted_text = pseudo_action(self._response_text(resp_obj))
                    submitted_dumps = call_dumps(submitted_text)
                    if forbidden_dumps and submitted_dumps:
                        return bool(forbidden_dumps & submitted_dumps)
                    return bool(
                        submitted_text
                        and _executed_action_key(submitted_text)
                        == _executed_action_key(forbidden.get("action", ""))
                    )

                def _candidate_exec_ok(candidate_actions, candidate_ok):
                    if not candidate_actions or not candidate_ok:
                        return False
                    if isinstance(candidate_actions[0], str):
                        compile_ok, _ = self._interface_check(candidate_actions[0])
                        return compile_ok
                    return True

                violated = _violates(response, actions)
                accepted = None
                rejected = False
                if violated:
                    harder = forbidden["context"] + (
                        "\n\n* The proposed action was rejected because it exactly "
                        "matched the blocked action. Return ONE different executable "
                        "action now; the blocked action will not be executed."
                    )
                    retry_msgs = self.prepare(instruction, obs, history, stall_context=harder)
                    retry_response = self._gen(retry_msgs)
                    logger.info("RESPONSE(stall-forbid): %s", retry_response)
                    retry_actions, retry_ok, retry_fp = self.execute(retry_response, obs)
                    accepted = bool(
                        _candidate_exec_ok(retry_actions, retry_ok)
                        and not _violates(retry_response, retry_actions)
                    )
                    if accepted:
                        response, actions = retry_response, retry_actions
                        interface_ok, fingerprint = retry_ok, retry_fp
                    else:
                        response, actions = retry_response, []
                        interface_ok, fingerprint = False, "stall_forbidden_rejected"
                        rejected = True
                logger.info("StallContract: %s", {
                    "violated": violated,
                    "regen_attempted": violated,
                    "regen_accepted": accepted,
                    "rejected": rejected,
                })
            except Exception as e:
                logger.warning("StallContract aborted (%s); rejecting unverified action", e)
                actions = []
                interface_ok, fingerprint = False, "stall_forbidden_verifier_error"

        # contents / Omni / traj.jsonl are text channels, but gen_func may return
        # list/dict content blocks (a dict would crash response[:800] below, a list
        # would nest into the next turn's text field via format_history). Textify
        # only non-str responses -- plain strings stay byte-identical to baseline.
        if not isinstance(response, str):
            response = self._response_text(response) or str(response)

        # update the contents
        self.contents.append(
            {
                "instruction": instruction,
                "index": len(self.contents),
                "response": response,
                "action": "Parse error" if not actions else actions[0],
                "exe_result": "Invalid action" if not actions else "",
                **obs,
            }
        )

        if self._omni is not None:
            text = f"Context: {obs.get('exe_result', '')[:200]}\nAction: {response[:800]}"
            try:
                result = self._omni.add_text(text, tags=[f"step:{self.contents[-1]['index']}"], force=True)
                logger.info(f"Omni store step={self.contents[-1]['index']} success={result.success} error={getattr(result, 'error', None)}")
            except Exception as e:
                logger.warning(f"Omni add_text failed: {e}")

        return response, actions

    def reset(self, _logger=None):
        global logger
        logger = _logger if _logger is not None else logging.getLogger("desktopenv.aguvis_agent")

        self.contents = []
        self._stall_forbidden = None
        self._init_omni()
        if self._stall_detector is not None:
            self._stall_detector.reset()

    def _init_omni(self):
        if not self._omni_data_dir:
            self._omni = None
            return
        from omni_memory import OmniMemoryOrchestrator, OmniMemoryConfig
        config = OmniMemoryConfig()
        config.embedding.model_name = "/home/c84445977/minilm"
        config.embedding.embedding_dim = 384
        config.llm.summary_model = self._omni_llm_model
        config.llm.query_model = self._omni_llm_model
        # OmniMemory's LLMConfig reads OPENAI_API_BASE (not OPENAI_BASE_URL) and
        # falls back to api.openai.com, so point it at the same endpoint the main
        # agent uses; otherwise internal LLM calls hang on the corporate proxy.
        base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url:
            config.llm.api_base_url = base_url
        self._omni = OmniMemoryOrchestrator(config=config, data_dir=self._omni_data_dir)
