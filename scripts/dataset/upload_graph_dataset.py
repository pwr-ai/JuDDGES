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
    commit_message: str = typer.Option("Add dataset", "--commit-message", "-m"),
    dry: bool = typer.Option(False, "--dry-run", "-d"),
):
    stats = _get_graph_info(root_dir)
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

    if not dry:
        api = HfApi()
        api.upload_folder(
            folder_path=str(root_dir),
            repo_id=repo_id,
            repo_type="dataset",
            ignore_patterns=["template_*", "metadata.yaml"],
            commit_message=commit_message,
        )


def _get_graph_info(root_dir: Path) -> dict[str, Any]:
    with open(root_dir / "data" / "judgment_graph.json") as file:
        g_data = json.load(file)

    g = nx.node_link_graph(g_data)
    assert nx.bipartite.is_bipartite(g)
    src_nodes, tgt_nodes = nx.bipartite.sets(g)

    attributes = _get_graph_attributes(g, src_nodes, tgt_nodes)
    stats = get_graph_stats(g, src_nodes, tgt_nodes)

    return attributes | stats


def _get_graph_attributes(
    graph: nx.Graph, src_nodes: set[int], tgt_nodes: set[int]
) -> dict[str, Any]:
    return {
        "judgment_attributes": ", ".join(
            sorted(f"`{attr}`" for attr in graph.nodes[src_nodes.pop()].keys())
        ),
        "legal_base_attributes": ", ".join(
            sorted(f"`{attr}`" for attr in graph.nodes[tgt_nodes.pop()].keys())
        ),
    }


def get_graph_stats(graph: nx.Graph, src_nodes: set[int], tgt_nodes: set[int]) -> dict[str, Any]:
    return {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "num_src_nodes": len(src_nodes),
        "num_target_nodes": len(tgt_nodes),
        "avg_degree": round(sum(dict(graph.degree()).values()) / graph.number_of_nodes(), 2),
    }


if __name__ == "__main__":
    typer.run(main)
