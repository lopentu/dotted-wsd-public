import inspect


def get_model_forward_params(model) -> list[str]:
    if hasattr(model, "module"):
        return get_model_forward_params(model.module)
    if hasattr(model, "model"):
        return get_model_forward_params(model.model)
    if hasattr(model, "forward"):
        return [p.name for p in inspect.signature(model.forward).parameters.values()]
    else:
        return []
