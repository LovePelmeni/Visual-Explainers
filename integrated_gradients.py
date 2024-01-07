import numpy 
import torch
from torch import nn
import typing
import cv2
import torchvision.models 

def compute_gradient(model: nn.Module, img: torch.Tensor, loss: typing.Callable, target: int):
    """
    Function returns
    """
    img.requires_grad = True
    logit = model(img)[0, target]
    logit.backward()
    img.requires_grad = False
    return img.grad.data.squeeze(0)
    
def normalize_gradient(img: torch.Tensor, percentile: float):
    if percentile > 1.0 or (percentile < 0):
        raise ValueError(msg='invalid percentile')
    return img / numpy.percentile(a=img.numpy(), q=percentile)

def get_baseline(input_img: numpy.ndarray, baseline_type, **kwargs):

    if baseline_type == 'black':
        return numpy.zeros_like(input_img).astype(numpy.uint8)

    if baseline_type == 'gaussian':
        return numpy.asarray(cv2.GaussianBlur(
            input_img, 
            sigmaX=kwargs.get("sigmaX", 2),
            sigmaY=kwargs.get("sigmaY", 2)
        ))
    raise ValueError('invalid baseline type.')

def integrated_gradients(
    network: nn.Module,
    input_img: numpy.ndarray, 
    actual_label: int,
    loss_function: nn.Module,
    n_steps: int, 
    baseline_type: typing.Literal['gaussian', 'black']
):
    baseline = torch.from_numpy(get_baseline(input_img, baseline_type)).permute(2, 0, 1).unsqueeze(0)
    input_img = torch.from_numpy(input_img.astype(numpy.uint8)).permute(2, 0, 1).unsqueeze(0)

    path = [baseline + alpha * (input_img - baseline) for alpha in numpy.linspace(0, 1, num=n_steps)]
    grads = [compute_gradient(model=network, img=img, loss=loss_function, target=actual_label) for img in path]
    norm_grads = [normalize_gradient(grad, percentile=0.98) for grad in grads[:-1]]
    int_grads = (input_img - baseline) * torch.cat(norm_grads[:-1]).mean(axis=0, keepdim=True)
    return int_grads.squeeze(0).permute(1, 2, 0)