"""Synchronize replicated LoRA parameters and accumulated gradients."""
import torch
import torch.distributed as dist


@torch.no_grad()
def sync_initial_parameters(parameters):
    for parameter in parameters:
        dist.broadcast(parameter, src=0)


@torch.no_grad()
def average_gradients(parameters):
    world_size = dist.get_world_size()
    active = torch.tensor([p.grad is not None for p in parameters],
                          device=parameters[0].device, dtype=torch.int32)
    dist.all_reduce(active)
    for parameter, ranks_with_grad in zip(parameters, active.tolist()):
        if not ranks_with_grad:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad)
        parameter.grad.div_(world_size)
