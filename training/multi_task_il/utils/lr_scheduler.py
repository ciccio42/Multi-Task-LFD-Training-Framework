import torch


def build_scheduler(optimizer, config):
    lr_schedule = config['type']
    print(f"Lr-schedule {lr_schedule}")
    if lr_schedule == 'ReduceLROnPlateau':
        return ReduceOnPlateau(optimizer)
    elif lr_schedule == 'ExponentialDecay':
        return ExponentialDecay(optimizer, **config)
    elif lr_schedule is None:
        return BaseScheduler(optimizer)
    else:
        raise NotImplementedError


class BaseScheduler:
    def __init__(self, optimizer):
        pass

    def step(self, **args):
        return


class ReduceOnPlateau(BaseScheduler):
    def __init__(self, optimizer):
        self._schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5)

    def step(self, val_loss, **kwargs):
        self._schedule.step(val_loss)


class ExponentialDecay(BaseScheduler):
    def __init__(self, optimizer, rate):
        assert 0 < rate < 1, "rate must be in (0, 1)"
        def lr_lambda(epoch): return rate ** epoch
        self._schedule = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda)

    def step(self, **kwargs):
        self._schedule.step()
