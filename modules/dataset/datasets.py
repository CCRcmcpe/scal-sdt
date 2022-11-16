import copy
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer


@dataclass
class Item:
    path: Path
    image: torch.Tensor | None
    token_ids: list[int]
    latent: torch.Tensor | None
    condition: torch.Tensor | None


class SDDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    # cached_conds = False
    # cached_latents = False

    def __init__(
            self,
            concepts,
            tokenizer: CLIPTokenizer,
            size=512,
            center_crop=False,
            pad_tokens=False,
            **kwargs
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.pad_tokens = pad_tokens

        self.entries = list[Item]()

        for concept in concepts:
            instance_entries = self.resolve_dataset(concept.instance_set)
            self.entries.extend(instance_entries)

        random.shuffle(self.entries)

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    @staticmethod
    def combine_prompt(prompt: str, txt_prompt: str, template: str):
        return template.replace("{PROMPT}", prompt).replace("{TXT_PROMPT}", txt_prompt)

    def tokenize(self, prompt: str) -> list[int]:
        return self.tokenizer(
            prompt,
            padding="max_length" if self.pad_tokens else "do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

    def resolve_dataset(self, dataset):
        for x in Path(dataset.path).iterdir():
            if not (x.is_file() and x.suffix != ".txt"):
                continue

            if dataset.combine_prompt_from_txt:
                content = x.with_suffix('.txt').read_text()
                prompt = SDDataset.combine_prompt(dataset.prompt, content, dataset.prompt_combine_template)
            else:
                prompt = dataset.prompt

            token_ids = self.tokenize(prompt)

            yield Item(x, None, token_ids, None, None)

    # def do_cache(self, vae: AutoencoderKL, text_encoder: CLIPWithSkip = None):
    #     train_dataloader = torch.utils.data.DataLoader(
    #         self, shuffle=True, collate_fn=lambda x: x, pin_memory=True
    #     )
    #
    #     with torch.inference_mode():
    #         for batch in tqdm(train_dataloader):
    #             for entry in batch:
    #                 entry.latent = vae.encode(entry.image).latent_dist.sample() * 0.18215
    #                 if text_encoder is not None:
    #                     entry.cond = text_encoder.forward(entry.token_ids)
    #
    #     self.cached_latents = True
    #     if text_encoder is not None:
    #         self.cached_conds = True

    def __len__(self):
        return len(self.entries)

    def _get_item(self, entries, index):
        entry = copy.copy(
            entries[index % len(entries)]
        )

        # if not self.cached_latents:
        image = self.read_img(entry.path)
        image = self.image_transforms(image)
        entry.image = image

        return entry

    def __getitem__(self, index) -> Item:
        return self._get_item(self.entries, index)

    @staticmethod
    def read_img(filepath: Path) -> Image:
        img = Image.open(filepath)

        if not img.mode == "RGB":
            img = img.convert("RGB")
        return img


class DBDataset(SDDataset):

    def __init__(self,
                 concepts,
                 tokenizer: CLIPTokenizer,
                 size=512,
                 center_crop=False,
                 pad_tokens=False,
                 **kwargs):
        super().__init__(concepts, tokenizer, size, center_crop, pad_tokens, **kwargs)

        self.class_entries = list[Item]()

        for concept in concepts:
            class_entries = self.resolve_dataset(concept.class_set)
            self.class_entries.extend(class_entries)

    def __getitem__(self, index) -> tuple[Item, Item]:
        instance = super().__getitem__(index)
        class_ = super()._get_item(self.class_entries, index)
        return instance, class_

    # def do_cache(self, vae: AutoencoderKL, text_encoder: CLIPWithSkip = None):
    #     train_dataloader = torch.utils.data.DataLoader(
    #         self, shuffle=True, collate_fn=lambda x: x, pin_memory=True
    #     )
    #
    #     with torch.inference_mode():
    #         for batch in tqdm(train_dataloader):
    #             for entries in batch:
    #                 for entry in entries:
    #                     entry.latent = vae.encode(entry.image).latent_dist.sample() * 0.18215
    #                     if text_encoder is not None:
    #                         entry.cond = text_encoder.forward(entry.token_ids)
    #
    #     self.cached_latents = True
    #     if text_encoder is not None:
    #         self.cached_conds = True


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example
