# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RoboLab benchmark tasks as an RLinf environment.

RoboLab (NVlabs/RoboLab) is an IsaacLab evaluation benchmark: 120 manipulation
tasks with automated success detection and no reward. This wrapper turns one
task into a vectorized RL environment for RLinf's embodied runners.

The simulator runs in a child process (``SubProcIsaacLabEnv``), which is also
where the task's ``SubtaskCompletionRecorderTerm`` lives, so the reward has to
be computed on that side. ``_ShapedRewardEnv`` is constructed inside the child
and replaces RoboLab's zero reward with the change in the recorder's progress
score plus a terminal bonus; the parent then sees an ordinary
``(obs, reward, terminated, truncated, info)`` step. Success is carried in
``info`` and read from there, never inferred from the reward.
"""

from __future__ import annotations

import torch

from rlinf.envs.sim.isaaclab.isaaclab_env import IsaaclabBaseEnv


class _ShapedRewardEnv:
    """Child-side wrapper over ``RobolabEnv`` that produces the shaped reward.

    ``reward_t = score_t - score_{t-1} + success_bonus * success_t`` where
    ``score`` is ``SubtaskStateMachine.get_total_score()`` in [0, 1]: the
    weighted fraction of the task's subtask stages completed, e.g. for a
    ``pick_and_place`` the stages grab, lift, move, drop, verify. A drop after a
    grab regresses the inner progress and yields a negative delta, which is
    intended. ``success`` is the task's own ``Terminations.success`` DoneTerm and
    is untouched, so a policy trained on this reward is still scored by the
    benchmark's rule.

    Frozen envs (already terminated within the episode) keep their last score,
    so their delta is 0 until reset, and ``terminated`` stays true on every
    frozen frame, so the bonus is paid once, on the frame success first
    appears. On that frame the recorder has already run its final step with
    ``is_complete()`` true, so the score is 1.0 on both sides and the delta is
    0 -- the bonus is the only signal there.
    """

    def __init__(self, env, success_bonus: float):
        from robolab.core.events.subtask_recorder import SubtaskCompletionRecorderTerm

        self.env = env
        self.success_bonus = float(success_bonus)
        self.num_envs = env.num_envs
        self.device = env.device
        rm = getattr(env, "recorder_manager", None)
        self._term = (
            rm.get_term(SubtaskCompletionRecorderTerm) if rm is not None else None
        )
        if self._term is None:
            raise RuntimeError(
                "RoboLab task has no SubtaskCompletionRecorderTerm; the shaped reward "
                "needs it. Check robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING and "
                "that the task defines `subtasks`."
            )
        self._prev_score = torch.zeros(self.num_envs, device=self.device)
        self._prev_success = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def _scores(self) -> torch.Tensor:
        return torch.tensor(
            [float(self._term.infos[i]["score"]) for i in range(self.num_envs)],
            device=self.device,
        )

    def reset(self, seed=None, env_ids=None):
        out = (
            self.env.reset(seed=seed)
            if env_ids is None
            else self.env.reset(seed=seed, env_ids=env_ids)
        )
        score = self._scores()
        if env_ids is None:
            self._prev_score = score
            self._prev_success[:] = False
        else:
            self._prev_score[env_ids] = score[env_ids]
            self._prev_success[env_ids] = False
        return out

    def step(self, action):
        obs, _zero_reward, terminated, truncated, info = self.env.step(action)
        score = self._scores()
        success = terminated.to(
            torch.bool
        )  # RobolabEnv: terminated == the success DoneTerm
        newly_succeeded = success & ~self._prev_success
        reward = (score - self._prev_score) + self.success_bonus * newly_succeeded.to(
            score.dtype
        )
        self._prev_score = score
        self._prev_success = success
        info = dict(info) if info is not None else {}
        info["success"] = success
        info["subtask_score"] = score
        return obs, reward, terminated, truncated, info

    def close(self):
        self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)


class RoboLabEnv(IsaaclabBaseEnv):
    """One RoboLab task, vectorized over ``num_envs``, for the RLinf env worker.

    Config surface (``env.train`` / ``env.eval``)::

        env_type: robolab
        init_params:
          id: MustardInLeftBinTask         # task class name from robolab/tasks/benchmark
          task_description: "Put the mustard in the left bin"
          success_bonus: 1.0
          instruction_type: default
        max_episode_steps: 900
        seed: 0

    The cameras are the DROID ``WRIST_LEFT_RIGHT_HEAD`` preset, which is what
    FlexPi was evaluated with; ``_wrap_obs`` composes them into the same single
    frame the RoboLab cosmos3 client sends (wrist on top, left|right at half
    resolution below) so the policy sees exactly its serving input.
    """

    def _make_env_function(self):
        task_name = self.isaaclab_env_id
        num_envs = self.cfg.init_params.num_envs
        seed = self.seed
        success_bonus = float(self.cfg.init_params.get("success_bonus", 1.0))
        instruction_type = str(self.cfg.init_params.get("instruction_type", "default"))

        def make_env_robolab():
            import os

            os.environ.pop("DISPLAY", None)  # headless; a stale DISPLAY breaks GLX
            from isaaclab.app import AppLauncher

            sim_app = AppLauncher(headless=True, enable_cameras=True).app

            from robolab.core.environments.runtime import create_env
            from robolab.registrations.droid.auto_env_registrations_jointpos import (
                auto_register_droid_envs,
            )
            from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD

            auto_register_droid_envs(task=[task_name], cameras=WRIST_LEFT_RIGHT_HEAD)
            env, _env_cfg = create_env(
                task_name,
                device="cuda:0",
                seed=seed,
                num_envs=num_envs,
                instruction_type=instruction_type,
                policy="rlinf",
            )
            return _ShapedRewardEnv(env, success_bonus), sim_app

        return make_env_robolab

    def _wrap_obs(self, obs):
        """RoboLab obs -> the dict FlexPi's ``predict_action_batch`` consumes.

        Mirrors ``policies/cosmos3/client.py::_pack_request``: one composite
        uint8 frame per env, ``[wrist ; left | right]`` with the two shoulder
        views at half resolution, plus proprio as ``[arm_joint_pos, gripper]``.
        """
        img = obs["image_obs"]
        wrist = img["wrist_cam"]  # [B, H, W, 3] uint8
        left = img["over_shoulder_left_camera"]
        right = img["over_shoulder_right_camera"]
        h, w = wrist.shape[1], wrist.shape[2]
        half = (h // 2, w // 2)

        def _down(x):
            x = x.permute(0, 3, 1, 2).float()
            x = torch.nn.functional.interpolate(
                x, size=half, mode="bilinear", align_corners=False
            )
            return x.permute(0, 2, 3, 1).to(wrist.dtype)

        bottom = torch.cat([_down(left), _down(right)], dim=2)  # [B, H/2, W, 3]
        composite = torch.cat([wrist, bottom], dim=1)  # [B, H + H/2, W, 3]

        prop = obs["proprio_obs"]
        states = torch.cat([prop["arm_joint_pos"], prop["gripper_pos"]], dim=-1)

        return {
            "main_images": composite,
            "task_descriptions": [self.task_description] * self.num_envs,
            "states": states,
            "wrist_images": wrist,
        }

    def _record_metrics(self, step_reward, terminations, infos):
        """Success comes from the task's DoneTerm, not from ``reward > 0``.

        The base class infers success from a positive reward, which the shaped
        reward would trigger on every subtask stage.
        """
        episode_info = {}
        self.returns += step_reward
        success = infos.get("success") if isinstance(infos, dict) else None
        if success is None:
            success = terminations
        self.success_once = (
            self.success_once | success.to(self.success_once.device).bool()
        )
        episode_info["success_once"] = self.success_once.clone()
        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = self.elapsed_steps.clone()
        episode_info["reward"] = episode_info["return"] / episode_info[
            "episode_len"
        ].clamp(min=1)
        if isinstance(infos, dict) and "subtask_score" in infos:
            episode_info["subtask_score"] = infos["subtask_score"].clone()
        infos["episode"] = episode_info
        return infos

    def step(self, actions=None, auto_reset=True):
        obs, step_reward, terminations, truncations, infos = self.env.step(actions)
        step_reward = step_reward.clone()
        terminations = terminations.clone()
        truncations = truncations.clone()
        obs = self._wrap_obs(obs)
        self._elapsed_steps += 1
        truncations = (self.elapsed_steps >= self.cfg.max_episode_steps) | truncations
        dones = terminations | truncations
        infos = self._record_metrics(
            step_reward, terminations, infos if isinstance(infos, dict) else {}
        )
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = terminations
            terminations[:] = False
        if dones.any() and auto_reset and self.auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)
        return obs, step_reward, terminations, truncations, infos
