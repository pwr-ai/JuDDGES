import json
from pathlib import Path
from typing import Any
import networkx as nx
from omegaconf import OmegaConf
import typer
from huggingface_hub import DatasetCard, DatasetCardData, HfApi

DATASET_CARD_TEMPLATE = "data/datasets/pl/graph/template_README.md"


def main(
    root_dir: Path = typer.Option(...),
    repo_id: str = typer.Option(...),
):
    stats = _get_stats(root_dir)
    config = OmegaConf.load(root_dir / "metadata.yaml")
    OmegaConf.resolve(config)

    card_data = DatasetCardData(
        language="pl",
        pretty_name="Polish Court Judgments Graph",
        size_categories="100K<n<1M",
        source_datasets=["pl-court-raw"],
        viewer=False,
        tags=["graph", "bipartite", "polish court"],
    )
    card = DatasetCard.from_template(
        card_data,
        template_path=DATASET_CARD_TEMPLATE,
        embedding_method=config["embeddings"]["embedding_model"]["name"],
        **stats,
    )
    card.save(root_dir / "README.md")

    api = HfApi()
    api.upload_folder(
        folder_path=str(root_dir),
        repo_id=repo_id,
        repo_type="dataset",
        ignore_patterns=["template_*", "metadata.yaml"],
    )


def _get_stats(root_dir: Path) -> dict[str, Any]:
    with open(root_dir / "data" / "judgement_graph.json") as file:
        g_data = json.load(file)

    g = nx.node_link_graph(g_data)
    src_nodes, tgt_nodes = nx.bipartite.sets(g)
    return {
        "num_nodes": g.number_of_nodes(),
        "num_edges": g.number_of_edges(),
        "num_src_nodes": len(src_nodes),
        "num_target_nodes": len(tgt_nodes),
        "avg_degree": sum(dict(g.degree()).values()) / g.number_of_nodes(),
    }


if __name__ == "__main__":
    typer.run(main)
