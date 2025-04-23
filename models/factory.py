from models.unet import UNet


def get_model(model_type: str):
    if model_type == "unet":
        return UNet()
    raise ValueError(f"Unknown model type: {model_type}")
