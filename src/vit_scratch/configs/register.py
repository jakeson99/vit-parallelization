from hydra_zen import store
from .model import ViT_tiny, ViT_small
from .data import CIFAR10
from .train import TrainConf
from .log import LogConf

# create groups so you can do model=vit_tiny, data=cifar10, etc.
st = store(group="model")
st.store(name="vit_tiny", node=ViT_tiny)
st.store(name="vit_small", node=ViT_small)

st = store(group="data")
st.store(name="cifar10", node=CIFAR10)

st = store(group="train")
st.store(name="base", node=TrainConf)

st = store(group="log")
st.store(name="base", node=LogConf)

# top-level composed config
from hydra_zen import make_config

TrainApp = make_config(
    model=None,  # filled by composition
    data=None,
    train=None,
    log=None,
)

# expose default composition
store().store(name="default", node=TrainApp)


def register_configs():
    # pushes everything into Hydra’s global store at import-time
    store().add_to_hydra_store()
