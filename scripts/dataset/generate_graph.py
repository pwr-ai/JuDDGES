import json
from pathlib import Path
from typing import Optional
import networkx as nx
import torch
import typer

from juddges.data.pl_court_graph import (
    create_judgement_legal_base_graph,
    create_judgement_legal_base_pyg_graph,
)


def main(
    dataset_dir: Path = typer.Option(...),
    embeddings_file: Optional[Path] = typer.Option(None),
    target_dir: Path = typer.Option(...),
):
    target_dir.mkdir(parents=True, exist_ok=True)

    g = create_judgement_legal_base_graph(dataset_dir)

    with open(target_dir / "judgement_graph.json", "w") as file:
        json.dump(nx.node_link_data(g), file, indent="\t")

    embs = torch.load(embeddings_file)
    pyg_data, index_2_iid = create_judgement_legal_base_pyg_graph(g, embs)
    pyg_dataset = {
        "data": pyg_data,
        "index_2_iid": index_2_iid,
    }
    torch.save(pyg_dataset, target_dir / "pyg_judgement_graph.pt")


if __name__ == "__main__":
    typer.run(main)
