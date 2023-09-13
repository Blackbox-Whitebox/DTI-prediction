from .pretraining import build as build_pretrain
from .transformer import build as build_transformer


def build(config):
    name = config["NAME"]
    if "pretrain" in name:
        net = build_pretrain(name, config)
    elif "transformer" in name:
        net = build_transformer(name, config)
    else:
        raise ValueError("Invalid `name`")

    return net

    