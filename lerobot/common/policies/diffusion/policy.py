import copy
import logging
import time
from collections import deque

import hydra
import torch
from diffusers.optimization import get_scheduler
from torch import Tensor, nn

from lerobot.common.policies.diffusion.model.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from lerobot.common.policies.utils import populate_queues
from lerobot.common.utils import get_safe_torch_device


class DiffusionPolicy(nn.Module):
    name = "diffusion"

    def __init__(
        self,
        cfg,
        cfg_device,
        cfg_noise_scheduler,
        cfg_optimizer,
        cfg_ema,
        shape_meta: dict,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        film_scale_modulation=True,
        **_,
    ):
        super().__init__()
        self.cfg = cfg
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps

        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None

        # TODO(now): In-house this.
        noise_scheduler = hydra.utils.instantiate(cfg_noise_scheduler)

        self.diffusion = DiffusionUnetImagePolicy(
            cfg,
            shape_meta=shape_meta,
            noise_scheduler=noise_scheduler,
            horizon=horizon,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            num_inference_steps=num_inference_steps,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            film_scale_modulation=film_scale_modulation,
        )

        self.device = get_safe_torch_device(cfg_device)
        self.diffusion.to(self.device)

        # TODO(alexander-soare): This should probably be managed outside of the policy class.
        self.ema_diffusion = None
        self.ema = None
        if self.cfg.use_ema:
            self.ema_diffusion = copy.deepcopy(self.diffusion)
            self.ema = hydra.utils.instantiate(
                cfg_ema,
                model=self.ema_diffusion,
            )

        self.optimizer = hydra.utils.instantiate(
            cfg_optimizer,
            params=self.diffusion.parameters(),
        )

        # TODO(rcadene): modify lr scheduler so that it doesnt depend on epochs but steps
        self.global_step = 0

        # configure lr scheduler
        self.lr_scheduler = get_scheduler(
            cfg.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.lr_warmup_steps,
            num_training_steps=cfg.offline_steps,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step - 1,
        )

    def reset(self):
        """
        Clear observation and action queues. Should be called on `env.reset()`
        """
        self._queues = {
            "observation.image": deque(maxlen=self.n_obs_steps),
            "observation.state": deque(maxlen=self.n_obs_steps),
            "action": deque(maxlen=self.n_action_steps),
        }

    def forward(self, batch: dict[str, Tensor], **_) -> Tensor:
        """A forward pass through the DNN part of this policy with optional loss computation."""
        return self.select_action(batch)

    @torch.no_grad
    def select_action(self, batch, **_):
        """
        Note: this uses the ema model weights if self.training == False, otherwise the non-ema model weights.
        # TODO(now): Handle a batch
        """
        assert "observation.image" in batch
        assert "observation.state" in batch
        assert len(batch) == 2  # TODO(now): Does this not have a batch dim?

        self._queues = populate_queues(self._queues, batch)

        if len(self._queues["action"]) == 0:
            # stack n latest observations from the queue
            batch = {key: torch.stack(list(self._queues[key]), dim=1) for key in batch}
            actions = self._generate_actions(batch)
            self._queues["action"].extend(actions.transpose(0, 1))

        action = self._queues["action"].popleft()
        return action

    def _generate_actions(self, batch):
        if not self.training and self.ema_diffusion is not None:
            return self.ema_diffusion.predict_action(batch)
        else:
            return self.diffusion.predict_action(batch)

    def update(self, batch, **_):
        """Run the model in train mode, compute the loss, and do an optimization step."""
        start_time = time.time()

        self.diffusion.train()

        loss = self.compute_loss(batch)

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.diffusion.parameters(),
            self.cfg.grad_clip_norm,
            error_if_nonfinite=False,
        )

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()

        if self.ema is not None:
            self.ema.step(self.diffusion)

        info = {
            "loss": loss.item(),
            "grad_norm": float(grad_norm),
            "lr": self.lr_scheduler.get_last_lr()[0],
            "update_s": time.time() - start_time,
        }

        return info

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        return self.diffusion.compute_loss(batch)

    def save(self, fp):
        torch.save(self.state_dict(), fp)

    def load(self, fp):
        d = torch.load(fp)
        missing_keys, unexpected_keys = self.load_state_dict(d, strict=False)
        if len(missing_keys) > 0:
            assert all(k.startswith("ema_diffusion.") for k in missing_keys)
            logging.warning(
                "DiffusionPolicy.load expected ema parameters in loaded state dict but none were found."
            )
        assert len(unexpected_keys) == 0
