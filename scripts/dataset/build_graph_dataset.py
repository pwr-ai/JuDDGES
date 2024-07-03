import json
from pathlib import Path

import networkx as nx
import torch
import typer
from omegaconf import OmegaConf

from juddges.data.pl_court_graph import (
    create_judgment_legal_base_graph,
    create_judgment_legal_base_pyg_graph,
)


def main(
    dataset_dir: Path = typer.Option(...),
    embeddings_root_dir: Path = typer.Option(None),
    target_dir: Path = typer.Option(...),
) -> None:
    data_dir = target_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    g = create_judgment_legal_base_graph(dataset_dir)

    with open(data_dir / "judgment_graph.json", "w") as file:
        json.dump(nx.node_link_data(g), file, indent="\t")

    embeddings_metadata_file = embeddings_root_dir / "all_embeddings/config.yaml"
    embs_conf = OmegaConf.load(embeddings_metadata_file)
    OmegaConf.save({"embeddings": embs_conf}, target_dir / "metadata.yaml")

    embeddings_file = embeddings_root_dir / "agg_embeddings.pt"
    embs = torch.load(embeddings_file)
    pyg_dataset = create_judgment_legal_base_pyg_graph(g, embs)
    torch.save(pyg_dataset, data_dir / "pyg_judgment_graph.pt")


if __name__ == "__main__":
    typer.run(main)
