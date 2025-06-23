import importlib


def get_model_class(cfg):
    module_name = cfg.model_architecture  # e.g., "simple_cnn"
    class_name = ''.join(part.capitalize() for part in module_name.split('_'))  # "SimpleCnn"

    module_path = f"custom_target_models.{module_name}"  # e.g., "custom_target_models.simple_cnn"
    module = importlib.import_module(module_path)

    model_class = getattr(module, class_name)
    return model_class
