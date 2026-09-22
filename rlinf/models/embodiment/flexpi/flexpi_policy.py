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

"""FlexPi as an RLinf policy, trained with DSRL.

FlexPi is a Wan2.2-TI2V-5B video DiT coupled to an ActionDiT that denoises a
32-step, 8-D action chunk (praxis-train, branch ``flexpi``). At serving time
only the action stream runs. DSRL leaves that model entirely frozen and trains
a small Gaussian policy over the *initial noise* the denoiser starts from, plus
the Q-heads that score it: the same shape RLinf already uses for OpenPI
(``rlinf/models/embodiment/openpi/openpi_action_model.py``, ``use_dsrl``), with
the same encoder and head modules.

The frozen model is built by importing Tao's serving class from his eval
directory read-only: it owns config composition, checkpoint loading, camera
composition and the normalizers, all of which were validated by the RoboLab
evaluations. The one thing FlexPi lacked -- taking the initial latent as an
argument -- is supplied by the ``latents_action`` overlay
(``/workspace/anjiangwei/flexpi-rl/patch_flexpi_latents.py``), which must be
first on ``PYTHONPATH`` in the interpreter this runs under.

Inference is batch-1 in the frozen model, so ``sample_actions`` loops over
envs. One inference is ~1.1 s on an H100; a decision step for 10 envs is ~11 s.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.compact_encoders import (
    CompactMultiQHead,
    CompactStateEncoder,
    LightweightImageEncoder64,
)
from rlinf.models.embodiment.modules.gaussian_policy import GaussianPolicy

# The heads are small enough to train in fp32; in bf16 Adam's 1e-4 steps are
# below the spacing of any weight above ~0.03 and the std head never learns.
_DSRL_DTYPE = torch.float32


def _load_serving_module(eval_dir: str):
    """Import Tao's ``server.py`` from his eval directory as a module."""
    path = Path(eval_dir) / "server.py"
    if not path.is_file():
        raise FileNotFoundError(f"FlexPi serving module not found: {path}")
    spec = importlib.util.spec_from_file_location("flexpi_serving", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["flexpi_serving"] = mod
    spec.loader.exec_module(mod)
    return mod


class _ValueHead(nn.Module):
    """V(obs) for PPO: its own encoders over the 64x64 view and proprio, then an MLP."""

    def __init__(self, state_dim, state_latent_dim, image_latent_dim, hidden_dims):
        super().__init__()
        self.image_encoder = LightweightImageEncoder64(
            num_images=1, latent_dim=image_latent_dim, image_size=64
        )
        self.state_encoder = CompactStateEncoder(
            state_dim=state_dim, hidden_dim=state_latent_dim
        )
        layers, d = [], state_latent_dim + image_latent_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        feat = torch.cat(
            [self.state_encoder(states), self.image_encoder(images)], dim=-1
        )
        return self.mlp(feat)  # [B, 1]


class FlexPiPolicy(BasePolicy, nn.Module):
    """Frozen FlexPi + a trainable noise policy over its initial action latent.

    Two training modes share the noise policy (a tanh-squashed Gaussian over
    the 32x8 latent) and the frozen model; the config picks the heads:

    * SAC / DSRL (``add_q_head: true``, ``use_dsrl: true``): Q-heads over
      (obs, noise), trained by ``EmbodiedSACFSDPPolicy`` from replay.
    * PPO (``add_value_head: true``): a value head over obs, trained by
      ``EmbodiedFSDPActor`` with GAE (``loss_type: actor_critic``). The
      rollout stores the pre-tanh sample and per-element log-probs, so the
      ratio is exact and RLinf's ``logprob_type`` aggregation applies.

    Config (``actor.model`` / ``rollout.model``)::

        model_type: flexpi
        precision: fp32
        is_lora: false
        use_dsrl: true          # SAC only: tells the SAC worker to train the noise policy
        add_q_head: true        # SAC heads; false for PPO
        add_value_head: false   # PPO value head; true for PPO
        num_action_chunks: 32   # FlexPi action horizon; the noise is [32, action_dim]
        action_dim: 8           # arm joints (7) + gripper (1)
        flexpi:
          eval_dir: /site/tao/cap100-v1-drop10-ft4-cfg5-20260911   # server.py + source/
          ckpt: .../step-18000/step_018000.pt
          dataset_stats: .../step-18000/step_018000.pt.stats.json
          task: cap100_v1_continue_textdrop10
          num_inference_steps: 20
          text_cfg_scale: 5.0
          seed: 42
        dsrl_state_dim: 8
        dsrl_image_latent_dim: 64
        dsrl_state_latent_dim: 64
        dsrl_hidden_dims: [128, 128, 128]
        dsrl_num_q_heads: 10

    Env obs contract (what ``RoboLabEnv._wrap_obs`` produces)::

        main_images: [B, H, W, 3] uint8   the composed canvas (wrist ; left|right)
        states:      [B, 8]              arm joints (7) + gripper (1), raw
        task_descriptions: list[str]
    """

    def __init__(
        self, cfg: DictConfig, torch_dtype: torch.dtype, device: str = "cuda:0"
    ):
        nn.Module.__init__(self)
        self.cfg = cfg
        # The latent shape is fixed by config so the actor, which trains the
        # heads without ever denoising, agrees with the rollout worker that
        # does. When the frozen model is loaded it must confirm both numbers.
        self.action_horizon = int(cfg.num_action_chunks)
        self.action_dim = int(cfg.action_dim)
        self.noise_dim = self.action_horizon * self.action_dim  # 256
        # The frozen 5B model is loaded on the first denoising request, so only
        # the rollout worker ever holds it: the SAC actor builds this class
        # twice (model + target) from the same config and never denoises.
        self._device = device
        self.serving = None

        # DSRL heads, wired exactly as OpenPI's use_dsrl path. The Q-head sees
        # the whole flattened latent, not a first timestep: with a 32x8 latent
        # the first row would be 3 % of what the noise policy chose.
        state_lat = int(cfg.get("dsrl_state_latent_dim", 64))
        image_lat = int(cfg.get("dsrl_image_latent_dim", 64))
        hidden = tuple(cfg.get("dsrl_hidden_dims", [128, 128, 128]))
        self.dsrl_state_dim = int(cfg.get("dsrl_state_dim", 8))
        self.dsrl_action_noise_net = GaussianPolicy(
            input_dim=state_lat + image_lat,
            output_dim=self.noise_dim,
            hidden_dims=hidden,
            low=None,
            high=None,
            action_horizon=1,
        ).to(dtype=_DSRL_DTYPE)
        self.actor_image_encoder = LightweightImageEncoder64(
            num_images=1, latent_dim=image_lat, image_size=64
        ).to(dtype=_DSRL_DTYPE)
        self.actor_state_encoder = CompactStateEncoder(
            state_dim=self.dsrl_state_dim, hidden_dim=state_lat
        ).to(dtype=_DSRL_DTYPE)
        self.add_q_head = bool(cfg.get("add_q_head", True))
        self.add_value_head = bool(cfg.get("add_value_head", False))
        if self.add_q_head:
            self.critic_image_encoder = LightweightImageEncoder64(
                num_images=1, latent_dim=image_lat, image_size=64
            ).to(dtype=_DSRL_DTYPE)
            self.critic_state_encoder = CompactStateEncoder(
                state_dim=self.dsrl_state_dim, hidden_dim=state_lat
            ).to(dtype=_DSRL_DTYPE)
            self.q_head = CompactMultiQHead(
                state_dim=state_lat,
                image_dim=image_lat,
                action_dim=self.noise_dim,
                hidden_dims=hidden,
                num_q_heads=int(cfg.get("dsrl_num_q_heads", 10)),
                output_dim=1,
            ).to(dtype=_DSRL_DTYPE)
        if self.add_value_head:
            self.value_head = _ValueHead(
                state_dim=self.dsrl_state_dim,
                state_latent_dim=state_lat,
                image_latent_dim=image_lat,
                hidden_dims=hidden,
            ).to(dtype=_DSRL_DTYPE)
        for name, module in self.named_modules():
            setattr(module, "_fsdp_wrap_name", name.split(".")[-1] if name else name)
        self.global_step = 0
        self.torch_compile_enabled = False

    def _load_serving(self, fp: DictConfig, device: str) -> None:
        """Load Tao's frozen FlexPi through his serving class (rollout side only)."""
        os.environ.setdefault("FLEXPI_DIR", str(Path(fp.eval_dir) / "source"))
        serving = _load_serving_module(str(fp.eval_dir))
        # Tao's class: config, checkpoint, processor/normalizers, camera layout.
        self.serving = serving.FlexPiPolicy(
            str(fp.ckpt),
            str(fp.task),
            int(fp.get("num_inference_steps", 20)),
            seed=fp.get("seed", 42),
            text_cfg_scale=float(fp.get("text_cfg_scale", 5.0)),
            dataset_stats=str(fp.dataset_stats),
            device=device,
        )
        model = self.serving.model
        # infer_action is decorated; inspect.signature follows __wrapped__.
        if "latents_action" not in inspect.signature(model.infer_action).parameters:
            raise RuntimeError(
                "FlexPi.infer_action has no `latents_action` argument: the DSRL overlay "
                "is not first on PYTHONPATH in this interpreter."
            )
        for p in model.parameters():
            p.requires_grad_(False)
        got = (int(self.serving.action_horizon), int(model.action_expert.action_dim))
        if got != (self.action_horizon, self.action_dim):
            raise ValueError(
                f"FlexPi checkpoint predicts {got[0]}x{got[1]} actions but config says "
                f"num_action_chunks={self.action_horizon}, action_dim={self.action_dim}"
            )

    def _require_serving(self):
        if self.serving is None:
            self._load_serving(self.cfg.flexpi, self._device)
            if bool(self.cfg.flexpi.get("warmup", True)) and torch.cuda.is_available():
                self._warmup()
        return self.serving

    def _warmup(self) -> None:
        """One throwaway inference so the first real action is not a cold call.

        The frozen model builds its glue cache, block-mask cache and
        torch.compile CUDA graphs on the first inference. Measured on the patch
        smoke: call #1 differed from a warm call on identical input by 0.26 in
        action units while warm calls were bit-identical. Doing it here also
        keeps ~100 s of compile out of the first rollout step.
        """
        h, w = self.serving.video_size
        obs = {
            "main_images": torch.zeros(1, h, w, 3, dtype=torch.uint8),
            "states": torch.zeros(1, self.dsrl_state_dim),
            "task_descriptions": ["warmup"],
        }
        self.sample_actions(obs, noise=torch.zeros(1, self.noise_dim))

    # ------------------------------------------------------------------ utils
    def set_global_step(self, global_step: int) -> None:
        self.global_step = global_step

    def _preprocess_dsrl_images(self, images, train: bool = False) -> torch.Tensor:
        """[B,H,W,3] uint8 (or [B,3,H,W]) -> [B,1,3,64,64] in [-1, 1]."""
        img = images[0] if isinstance(images, list) else images
        if img.shape[-1] == 3:
            img = img.permute(0, 3, 1, 2)
        img = img.float()
        if img.max() > 1.0:
            img = img / 255.0
        img = torch.nn.functional.interpolate(
            img, size=(64, 64), mode="bilinear", align_corners=False
        )
        img = img * 2.0 - 1.0
        return img.unsqueeze(1)

    def _preprocess_states(self, states: torch.Tensor) -> torch.Tensor:
        if states.dim() > 2:
            states = states.reshape(states.shape[0], -1)
        return states.to(dtype=_DSRL_DTYPE)

    @staticmethod
    def _as_internal(obs: dict) -> dict:
        """Pick the image the DSRL encoders see: the env's 64x64 view when present.

        ``extra_view_images`` [B, 1, 64, 64, 3] is what RoboLabEnv provides and
        what replay transitions keep; ``main_images`` is the fallback so the
        encoders also work on a raw composite.
        """
        if "images" in obs:
            return obs
        small = obs.get("extra_view_images")
        if small is not None:
            return {"images": [small[:, 0]], "states": obs["states"]}
        if "main_images" in obs:
            return {"images": [obs["main_images"]], "states": obs["states"]}
        raise ValueError(
            f"Invalid obs format: {list(obs)}. "
            "Expected 'images', 'extra_view_images' or 'main_images'."
        )

    # ---------------------------------------------------------------- DSRL
    def sac_forward(
        self, obs=None, data=None, train=False, return_dist_params=False, **kwargs
    ):
        """Noise policy: obs -> (noise [B, noise_dim], logprob [B], dist_params)."""
        if obs is None:
            obs = data.get("obs", data) if data is not None else kwargs.get("obs", {})
        obs = self._as_internal(obs)
        device = next(self.actor_image_encoder.parameters()).device
        images = self._preprocess_dsrl_images(obs["images"], train=train).to(
            device=device, dtype=_DSRL_DTYPE
        )
        states = self._preprocess_states(obs["states"]).to(device=device)
        features = torch.cat(
            [self.actor_state_encoder(states), self.actor_image_encoder(images)], dim=-1
        )
        deterministic = kwargs.get("mode", "train") == "eval"
        noise, logprobs = self.dsrl_action_noise_net.sample(
            features, deterministic=deterministic
        )
        noise = noise.reshape(noise.shape[0], -1)
        dist_params = None
        if return_dist_params:
            dist = self.dsrl_action_noise_net.forward(features)
            dist_params = (dist.mean, dist.stddev)
        return noise, logprobs, dist_params

    def sac_q_forward(
        self,
        obs=None,
        data=None,
        actions=None,
        detach_encoder=False,
        train=False,
        **kwargs,
    ):
        """Q-heads: (obs, noise) -> [B, num_q_heads]."""
        if obs is None:
            obs = data.get("obs", data) if data is not None else kwargs.get("obs", {})
        if actions is None:
            actions = kwargs.get("actions")
        obs = self._as_internal(obs)
        device = next(self.critic_image_encoder.parameters()).device
        images = self._preprocess_dsrl_images(obs["images"], train=train).to(
            device=device, dtype=_DSRL_DTYPE
        )
        states = self._preprocess_states(obs["states"]).to(device=device)
        actions = actions.reshape(actions.shape[0], -1).to(
            device=device, dtype=_DSRL_DTYPE
        )
        image_features = self.critic_image_encoder(images)
        state_features = self.critic_state_encoder(states)
        if detach_encoder:
            image_features = image_features.detach()
            state_features = state_features.detach()
        return self.q_head(state_features, image_features, actions)

    # ----------------------------------------------------- frozen inference
    @torch.no_grad()
    def sample_actions(
        self, env_obs: dict, noise: Optional[torch.Tensor] = None
    ) -> dict:
        """Denoise one action chunk per env from ``noise`` ([B, noise_dim]).

        Returns ``actions`` [B, horizon, 8] in the robot's units (denormalized),
        exactly what the serving path returns to the RoboLab client.
        """
        s = self._require_serving()
        canvases = env_obs["main_images"]
        states = env_obs["states"]
        prompts = env_obs.get("task_descriptions")
        if prompts is None or len(prompts) != canvases.shape[0]:
            summary = {
                k: (
                    tuple(v.shape)
                    if torch.is_tensor(v)
                    else (type(v).__name__, len(v) if hasattr(v, "__len__") else None)
                )
                for k, v in env_obs.items()
            }
            raise ValueError(
                "FlexPi needs one task description per env, got "
                f"{None if prompts is None else len(prompts)} for {canvases.shape[0]} envs; obs: {summary}"
            )
        B = canvases.shape[0]
        if noise is None:
            noise = torch.randn(B, self.noise_dim)
        out = []
        for b in range(B):
            canvas_u8 = canvases[b].detach().to("cpu").numpy().astype(np.uint8)
            cams = s._split_composed(canvas_u8)
            per_cam = s._per_cam(cams)
            canvas = s._compose(per_cam)
            state8 = states[b].detach().to("cpu").float().numpy().astype(np.float32)
            proprio = s._norm_proprio(state8)
            z = (
                noise[b]
                .detach()
                .to(torch.float32)
                .reshape(1, self.action_horizon, self.action_dim)
            )
            pred = s.model.infer_action(
                prompt=str(prompts[b]),
                input_image=canvas,
                per_cam=per_cam,
                action_horizon=s.action_horizon,
                num_video_frames=s.num_video_frames,
                proprio=proprio,
                negative_prompt="",
                text_cfg_scale=s.text_cfg_scale,
                num_inference_steps=s.num_inference_steps,
                # A fixed seed together with the injected latent. FlexPi has
                # seven noise-init sites; the overlay gates the action latent
                # and the seeded generator drives the rest, so the action is a
                # deterministic function of z. Measured: same z with seed=None
                # differed by 0.26, nearly the z1-vs-z2 gap.
                seed=self.serving._seed,
                joint_video=s.joint_video,
                joint_dino=s.joint_dino,
                joint_pointmap=s.joint_pointmap,
                return_stream_latents=False,
                latents_action=z,
            )
            out.append(
                torch.from_numpy(s._denorm_action(pred["action"]).astype(np.float32))
            )
        actions = torch.stack(out, dim=0)  # [B, 32, 8]
        return {
            "actions": actions,
            "denoise_inds": None,
            "prev_logprobs": None,
            "prev_values": None,
        }

    # ------------------------------------------------------- BasePolicy API
    # ----------------------------------------------------------------- PPO
    def _encoded_obs(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """(images [B,1,3,64,64] in [-1,1], states [B,8]) on the heads' device."""
        obs = self._as_internal(obs)
        device = next(self.actor_image_encoder.parameters()).device
        images = self._preprocess_dsrl_images(obs["images"]).to(
            device=device, dtype=_DSRL_DTYPE
        )
        states = self._preprocess_states(obs["states"]).to(device=device)
        return images, states

    def _noise_normal(self, images, states) -> torch.distributions.Normal:
        """The pre-tanh Gaussian of the noise policy, elementwise [B, noise_dim]."""
        features = torch.cat(
            [self.actor_state_encoder(states), self.actor_image_encoder(images)], dim=-1
        )
        dist = self.dsrl_action_noise_net.forward(features)
        # SquashedNormal = tanh(Independent(Normal)); the Normal is what PPO needs.
        return dist.base_dist.base_dist

    @staticmethod
    def _tanh_logprobs(normal, pre_tanh: torch.Tensor) -> torch.Tensor:
        """Per-element log-prob of z = tanh(u) under the squashed Gaussian."""
        z = torch.tanh(pre_tanh)
        return normal.log_prob(pre_tanh) - torch.log(1 - z.pow(2) + 1e-7)

    def default_forward(
        self,
        forward_inputs,
        compute_logprobs=True,
        compute_entropy=False,
        compute_values=False,
        **kwargs,
    ):
        """PPO recompute on stored rollout inputs.

        Returns per-element ``logprobs`` [B, noise_dim] (``logprob_type``
        decides the aggregation), Gaussian ``entropy`` per element, and
        ``values`` [B, 1] from the value head.
        """
        images, states = self._encoded_obs(
            {
                "extra_view_images": forward_inputs["extra_view_images"],
                "states": forward_inputs["states"],
            }
        )
        normal = self._noise_normal(images, states)
        pre_tanh = forward_inputs["pre_tanh"].to(
            device=images.device, dtype=_DSRL_DTYPE
        )
        out = {"logprobs": self._tanh_logprobs(normal, pre_tanh)}
        if compute_entropy:
            out["entropy"] = normal.entropy()
        if compute_values:
            out["values"] = self.value_head(images, states)
        return out

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SAC:
            return self.sac_forward(**kwargs)
        if forward_type == ForwardType.SAC_Q:
            return self.sac_q_forward(**kwargs)
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError(f"FlexPiPolicy has no forward for {forward_type}")

    def predict_action_batch(self, env_obs=None, mode: str = "train", **kwargs):
        """Rollout step: the noise policy samples z, FlexPi denoises it.

        ``actions`` go to the env; ``forward_inputs["action"]`` is the NOISE,
        which is what RL trains on. ``model_action`` keeps the denoised chunk.
        """
        if self.add_value_head:
            return self._predict_ppo(env_obs, mode)
        noise, noise_logprob, _ = self.sac_forward(env_obs, train=False, mode=mode)
        outputs = self.sample_actions(env_obs, noise=noise)
        actions = outputs["actions"]
        B = actions.shape[0]
        forward_inputs = {
            "action": noise.detach().to(torch.float32).reshape(B, -1).contiguous(),
            "model_action": actions.reshape(B, -1).contiguous(),
            "denoise_inds": None,
        }
        result = {
            "forward_inputs": forward_inputs,
            "prev_logprobs": noise_logprob.detach().to(torch.float32),
            # SAC bootstraps a truncated chunk from next_obs inside its TD
            # target, so the reward-level bootstrap the env worker adds on
            # truncation must be 0 or the value is counted twice.
            "prev_values": torch.zeros(B, 1, dtype=torch.float32),
        }
        return actions, result

    @torch.no_grad()
    def _predict_ppo(self, env_obs: dict, mode: str):
        images, states = self._encoded_obs(env_obs)
        normal = self._noise_normal(images, states)
        pre_tanh = normal.loc if mode == "eval" else normal.rsample()
        noise = torch.tanh(pre_tanh)
        logprobs = self._tanh_logprobs(normal, pre_tanh)  # [B, noise_dim]
        values = self.value_head(images, states)  # [B, 1]
        outputs = self.sample_actions(env_obs, noise=noise.to(torch.float32))
        actions = outputs["actions"]
        B = actions.shape[0]
        forward_inputs = {
            "action": noise.detach().to(torch.float32).reshape(B, -1).contiguous(),
            "pre_tanh": pre_tanh.detach().to(torch.float32).contiguous(),
            "model_action": actions.reshape(B, -1).contiguous(),
            # What default_forward needs to recompute log-probs and values.
            "extra_view_images": env_obs["extra_view_images"],
            "states": env_obs["states"],
        }
        result = {
            "forward_inputs": forward_inputs,
            "prev_logprobs": logprobs.detach().to(torch.float32).contiguous(),
            "prev_values": values.detach().to(torch.float32).contiguous(),
        }
        return actions, result


def get_model(cfg: DictConfig, torch_dtype: torch.dtype) -> FlexPiPolicy:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return FlexPiPolicy(cfg, torch_dtype, device=device)
