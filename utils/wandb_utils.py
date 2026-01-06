def init_wandb(args, name=None, group=None, tags=None):
    if not getattr(args, 'use_wandb', False):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb is not installed; run `pip install wandb`") from exc

    wandb.init(
        project=getattr(args, 'wandb_project', None),
        entity=getattr(args, 'wandb_entity', None),
        name=name,
        group=group,
        tags=tags,
        mode=getattr(args, 'wandb_mode', None),
        config=vars(args)
    )
    return wandb.run


def log_wandb(metrics, step=None):
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    wandb.log(metrics, step=step)


def finish_wandb():
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is not None:
        wandb.finish()
