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

from collections import deque

import torch
from omegaconf import OmegaConf, open_dict

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

    RoboLab is an evaluation benchmark: a terminated env is *frozen* (zero
    action, state held, ``terminated`` stays true) rather than reset, and its
    ``_reset_idx`` skips frozen envs even on an explicit reset. Two things
    follow. Frozen frames get reward 0, so the bonus is paid once, on the
    frame success first appears, and score drift on a held scene is ignored.
    And ``reset(env_ids)`` performs a real reset of those envs, so RLinf's
    chunk-end auto-reset starts a new episode.

    The score is read from ``term.infos``, which the recorder writes in
    ``record_post_step`` before IsaacLab's in-step reset logic runs, so on the
    success frame it is the finished episode's 1.0.
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

    def _scores(self) -> torch.Tensor:
        return torch.tensor(
            [float(self._term.infos[i]["score"]) for i in range(self.num_envs)],
            device=self.device,
        )

    def reset(self, seed=None, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device)
        env = self.env
        # RobolabEnv._reset_idx freezes every env it is handed once stepping
        # has begun, so an explicit reset must take its initial-reset path
        # (a real IsaacLab reset plus predicate state), then clear the frozen
        # bookkeeping that reset_eval_state() clears for all envs.
        env._has_stepped = False
        try:
            obs, info = env.reset(seed=seed, env_ids=env_ids)
        finally:
            env._has_stepped = True
        out = (self._with_instruction(obs), info)
        env._frozen_envs[env_ids] = False
        env._pre_step_frozen[env_ids] = False
        for eid in env_ids.tolist():
            env._env_results.pop(eid, None)
            env._env_term_step.pop(eid, None)
        self._prev_score[env_ids] = 0.0  # the state machines were reset
        return out

    def _with_instruction(self, obs):
        # The benchmark's own resolved instruction (task class + instruction_type)
        # is the prompt the policy was evaluated with; carry it with the obs.
        obs = dict(obs)
        obs["instruction"] = str(self.env.cfg.instruction)
        return obs

    def step(self, action):
        obs, _zero_reward, terminated, truncated, info = self.env.step(action)
        obs = self._with_instruction(obs)
        # Snapshot RoboLab took before this step: envs that held state.
        frozen = self.env._pre_step_frozen
        # An env RoboLab reset itself during this step (a termination within
        # its first two frames is treated as a physics artifact) starts over.
        restarted = self.env.episode_length_buf == 0
        score = self._scores()
        success = terminated.to(torch.bool)
        reward = (score - self._prev_score) + self.success_bonus * success.to(
            score.dtype
        )
        reward = torch.where(frozen | restarted, torch.zeros_like(reward), reward)
        self._prev_score = torch.where(restarted, torch.zeros_like(score), score)
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
          task_description: "Put the mustard in the left bin"   # fallback; the benchmark's own text is used
          success_bonus: 1.0
          instruction_type: default
        max_episode_steps: 900
        seed: 0
        override_cfgs:                     # optional: one entry per env-worker rank
          - init_params: {id: BowlInBinTask}
          - init_params: {id: ReorientRedMugTask}

    With ``override_cfgs`` the env worker gives rank ``i`` its entry as
    ``cfg.override_cfg``, which is merged over this config before the task is
    created, so one job can train several tasks with one env worker each.

    The cameras are the DROID ``WRIST_LEFT_RIGHT_HEAD`` preset, which is what
    FlexPi was evaluated with; ``_wrap_obs`` composes them into the same single
    frame the RoboLab cosmos3 client sends (wrist on top, left|right at half
    resolution below) so the policy sees exactly its serving input.
    """

    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        override = cfg.get("override_cfg", None)
        if override:
            with open_dict(cfg):
                cfg = OmegaConf.merge(cfg, override)
        super().__init__(cfg, num_envs, seed_offset, total_num_processes, worker_info)

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
        # A 64x64 copy of the composite for the DSRL encoders. It is the only
        # image the actor needs, so replay transitions can keep just this view
        # (rollout.transition_obs_keys) instead of the full-resolution frames.
        small = torch.nn.functional.interpolate(
            composite.permute(0, 3, 1, 2).float(),
            size=(64, 64),
            mode="bilinear",
            align_corners=False,
        )
        small = (
            small.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).unsqueeze(1)
        )

        instruction = obs.get("instruction", self.task_description)
        return {
            "main_images": composite,
            "task_descriptions": [instruction] * self.num_envs,
            "states": states,
            "wrist_images": wrist,
            "extra_view_images": small,  # [B, 1, 64, 64, 3]
        }

    def _init_metrics(self):
        super()._init_metrics()
        # Per-episode outcomes, counted once when an episode first ends, for
        # rates that are not biased toward the episodes that happen to finish
        # inside one rollout step (early on those are only the successes:
        # failures run until the task's time limit).
        self._episode_done = torch.zeros(self.num_envs, dtype=torch.bool).to(
            self.device
        )
        self._recent_outcomes: deque[bool] = deque(maxlen=100)
        self._finished = 0
        self._succeeded = 0

    def _reset_metrics(self, env_idx=None):
        super()._reset_metrics(env_idx)
        if env_idx is None:
            self._episode_done[:] = False
        else:
            self._episode_done[env_idx] = False

    def _count_outcomes(self, terminations, truncations):
        newly_done = (terminations | truncations) & ~self._episode_done
        if newly_done.any():
            for ok in (self.success_once | terminations)[newly_done].tolist():
                self._recent_outcomes.append(bool(ok))
                self._finished += 1
                self._succeeded += int(ok)
            self._episode_done |= newly_done

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
        # One env worker runs one task, so a task-named copy of the success
        # metric gives per-task curves when several tasks train in one job.
        task = self.isaaclab_env_id
        episode_info[f"success_once/{task}"] = self.success_once.clone()
        # Rates over finished episodes: the last 100 and all so far. Constant
        # across envs, so the mean the runner logs is the rate itself.
        if self._recent_outcomes:
            recent = sum(self._recent_outcomes) / len(self._recent_outcomes)
            overall = self._succeeded / self._finished
            episode_info[f"success_rate_100/{task}"] = torch.full_like(
                self.returns, recent
            )
            episode_info[f"success_rate_all/{task}"] = torch.full_like(
                self.returns, overall
            )
            episode_info[f"episodes_finished/{task}"] = torch.full_like(
                self.returns, float(self._finished)
            )
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
        infos = infos if isinstance(infos, dict) else {}
        success = infos.get("success")
        if success is not None:
            self.success_once = (
                self.success_once | success.to(self.success_once.device).bool()
            )
        self._count_outcomes(terminations, truncations)
        infos = self._record_metrics(step_reward, terminations, infos)
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = terminations
            terminations[:] = False
        if dones.any() and auto_reset and self.auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)
        return obs, step_reward, terminations, truncations, infos
