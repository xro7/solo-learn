import torch


class DinoVisionTransformer:

    # https://github.com/facebookresearch/dinov2/tree/main
    def __init__(self):
        pass

    @classmethod
    def from_hub(cls, model_name='dinov2_vitl14'):
        return torch.hub.load('facebookresearch/dinov2', model_name)


def vit_dinov2_large(model_name='dinov2_vitl14'):
    model = DinoVisionTransformer.from_hub(model_name=model_name)
    model.num_features = 1024
    return model
