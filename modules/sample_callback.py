from os import PathLike
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from PIL import Image
from pytorch_lightning.utilities import rank_zero_only
from tqdm import tqdm

from .model import StableDiffusionModel
from .utils import rename_keys


class SampleCallback(pl.Callback):
    _CONFIG_TRANSFORM = {"cfg_scale": "guidance_scale", "steps": "num_inference_steps",
            "num_samples": None, "seed": None}

    def __init__(self, sample_save_dir: str | PathLike):
        self.sample_dir = Path(sample_save_dir)

    @torch.inference_mode()
    @rank_zero_only
    def on_train_batch_end(self, trainer: pl.Trainer, model: StableDiffusionModel, outputs,
                           batch: Any, batch_idx: int) -> None:
        sampling_config = model.config.get("sampling")
        global_step = trainer.global_step

        if (sampling_config is None or
                not any(sampling_config.concepts) or
                global_step % sampling_config.interval_steps != 0):
            return

        batch_size = sampling_config.batch_size

        save_dir = self.sample_dir / str(global_step)
        save_dir.mkdir(parents=True, exist_ok=True)

        samples = dict[str, list[Image.Image]]()

        text_encoder_training = model.text_encoder.training
        model.text_encoder.eval()

        for concept in tqdm(sampling_config.concepts, unit="concept"):
            generator = torch.Generator(device=model.pipeline.device).manual_seed(concept.seed)

            concept_samples = list[Image.Image]()
            i = concept.num_samples
            with tqdm(total=concept.num_samples + (concept.num_samples % batch_size),
                      desc="Generating samples") as progress:
                while True:
                    actual_bsz = i if i - batch_size < 0 else batch_size

                    if actual_bsz <= 0:
                        break

                    concept_samples.extend(
                        model.pipeline(
                            num_images_per_prompt=actual_bsz,
                            generator=generator,
                            **rename_keys(concept, self._CONFIG_TRANSFORM)
                        ).images
                    )
                    progress.update(actual_bsz)

                    i -= actual_bsz
            samples[concept.prompt] = concept_samples

        model.text_encoder.train(text_encoder_training)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for i, (_, images) in enumerate(samples.items()):
            for j, image in enumerate(images):
                image.save(save_dir / f"{i}-{j}.png")

        wandb_config = model.config.loggers.get("wandb")

        if (wandb_config is not None and
                wandb_config.get("sample", False) and
                any(samples)):
            import wandb
            log_samples = {
                "samples": {
                    prompt: [wandb.Image(x) for x in images] for prompt, images in samples.items()
                }
            }
            wandb.log(log_samples, global_step)
