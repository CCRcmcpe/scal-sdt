import itertools
from collections.abc import Iterable
from pathlib import Path

import PIL.Image as Image


def list_images(*paths: Path) -> Iterable[Path]:
    return itertools.chain(*((
        x for x in path.iterdir() if
        x.is_file() and
        x.suffix.lower() != ".txt" and
        x.suffix.lower() in Image.registered_extensions().keys()
    ) for path in paths))


def read_image(filepath: Path) -> Image.Image:
    img = Image.open(filepath)

    if not img.mode == "RGB":
        img = img.convert("RGB")
    return img


def rename_keys(source: dict, key_dict: dict):
    return {key_dict.get(k, k): v for k, v in source.items() if key_dict.get(k, "") != None}


def get_class(name: str):
    import importlib
    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name, package=None)
    return getattr(module, class_name)


def physical_core_count():
    import psutil
    return psutil.cpu_count(logical=False)
