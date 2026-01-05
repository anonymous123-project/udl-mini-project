from registry.registry import LOSSES, MODELS, ACQUISITION_FUNCTIONS

import torch.optim as optim


def build_loss(cfg):
    assert isinstance(cfg, dict)
    assert "type" in cfg
    loss_type = cfg["type"]
    kwargs = {k: v for k, v in cfg.items() if k != "type"}

    return LOSSES.get(loss_type)(**kwargs)


def build_model(cfg):
    assert isinstance(cfg, dict)
    assert "type" in cfg

    model_type = cfg["type"]
    kwargs = {k: v for k, v in cfg.items() if k != "type"}

    return MODELS.get(model_type)(**kwargs)


def build_acquisiton_fn(cfg):
    assert isinstance(cfg, dict)
    assert "type" in cfg

    model_type = cfg["type"]
    kwargs = {k: v for k, v in cfg.items() if k != "type"}

    return ACQUISITION_FUNCTIONS.get(model_type)(**kwargs)


def build_optimizer(params, cfg):
    assert isinstance(cfg, dict)
    assert "type" in cfg

    opt_type = cfg["type"]
    kwargs = {k: v for k, v in cfg.items() if k != "type"}

    if opt_type == "Adam":
        return optim.Adam(params, **kwargs)

    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")
