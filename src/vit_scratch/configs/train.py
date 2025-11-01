from hydra_zen import make_config

TrainConf = make_config(
    epochs=100,
    lr=3e-4,
    wd=0.05,
    warmup_epochs=5,
    amp=True,
    grad_clip=1.0,
    seed=967854,
    ddp=False,
    fsdp=False,
    flas_attn=False,
    global_batch=None,
)
