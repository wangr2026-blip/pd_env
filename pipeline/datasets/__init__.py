"""Dataset registry. Add a corpus (other language / task) by registering a Dataset subclass."""


def get_dataset(cfg):
    dataset_cfg = cfg["dataset"]
    name = dataset_cfg["name"]
    if name == "synthetic":
        from pipeline.datasets.synthetic import SyntheticDataset
        return SyntheticDataset(dataset_cfg, cfg["seed"])
    if name == "mpower":
        from pipeline.datasets.mpower import MPowerDataset
        return MPowerDataset(dataset_cfg, cfg["seed"])
    raise ValueError(f"Unknown dataset: {name}")
