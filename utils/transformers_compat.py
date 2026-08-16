# Transformers auto-class compatibility for the local QwenVL nodes
#
# transformers 5.x removed AutoModelForVision2Seq. Its replacements are
# AutoModelForImageTextToText and - for the newest checkpoints such as
# Qwen3.5 / Qwen3.8 - AutoModelForMultimodalLM. transformers 4.x only ships the
# older two, so resolve whichever classes the running version actually has and
# try them in newest-first order.

import transformers

_AUTO_CLASS_NAMES = (
    "AutoModelForMultimodalLM",
    "AutoModelForImageTextToText",
    "AutoModelForVision2Seq",
)


def _auto_classes():
    classes = [
        getattr(transformers, name)
        for name in _AUTO_CLASS_NAMES
        if hasattr(transformers, name)
    ]
    if not classes:
        raise ImportError(
            "transformers is too old for the QwenVL nodes - none of "
            f"{', '.join(_AUTO_CLASS_NAMES)} exist. Upgrade with: "
            "pip install -U transformers"
        )
    return classes


def load_vision_model(model_path, **load_kwargs):
    """from_pretrained through the first auto class that maps this architecture."""
    last_error = None
    for auto_class in _auto_classes():
        try:
            return auto_class.from_pretrained(model_path, **load_kwargs)
        except ValueError as error:
            # This auto class does not know the architecture - try the next one.
            last_error = error
    raise last_error
