from hydra_zen import make_config

LogConf = make_config(
    out_dir="runs/cifar_vit_tiny",
    project="vit-from-scratch",
    notes="baseline",
    log_interval=50,
    save_every=1,
)
