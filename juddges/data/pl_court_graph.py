from pathlib import Path
import random
import polars as pl
import networkx as nx
from tqdm.auto import tqdm

FEATURES = ["_id", "signature", "date", "court_name", "department_name", "isap_ids"]


def create_judgement_legal_base_graph(
    dataset_root: Path, verbose: bool = True, sample_size: int | None = None
) -> nx.Graph:
    """Create networkx graph where node are judgement and legal bases, both connected by edges

    Args:
        dataset_root (Path): Path to directory with parquet files containing raw dataset

    Returns:
        nx.Graph: graph where node are judgement and legal bases, both connected by edges
    """
    raw_dataset = pl.scan_parquet(str(dataset_root / "*.parquet"))
    dataset = filter_data_and_extract_isap_ids(raw_dataset, sample_size)
    graph = create_bipartite_graph(dataset, verbose=verbose)

    return graph


def filter_data_and_extract_isap_ids(
    dataset: pl.LazyFrame, sample_size: int | None
) -> pl.DataFrame:
    if sample_size:
        random_idx = random.sample(range(len(dataset)), sample_size)
        dataset = dataset.select(random_idx)

    return (
        dataset.filter(pl.col("text_legal_bases").list.len() > 0)
        .with_columns(
            pl.col("text_legal_bases")
            .map_elements(lambda legal_bases: list(set(lb["isap_id"] for lb in legal_bases)))
            .alias("isap_ids")
        )
        .select(FEATURES)
        .collect()
    )


def create_bipartite_graph(dataset: pl.DataFrame, verbose: bool) -> nx.Graph:
    isap_ids = dataset["isap_ids"].explode().unique()
    isap_id_2_index = {iid: index for index, iid in enumerate(isap_ids, start=len(dataset))}

    source_nodes = list(range(len(dataset)))
    target_nodes = list(isap_id_2_index.values())
    edge_list = []
    attrs = {}

    for index, doc in tqdm(
        enumerate(dataset.iter_rows(named=True)), total=len(dataset), disable=not verbose
    ):
        attrs[index] = {
            "_id": doc["_id"],
            "signature": doc["signature"],
            "date": doc["date"],
            "court_name": doc["court_name"],
            "department_name": doc["department_name"],
        }
        target_idx = [isap_id_2_index[iid] for iid in doc["isap_ids"]]
        edge_list.extend([(index, t_idx) for t_idx in target_idx])

    g = nx.Graph()
    g.add_nodes_from(source_nodes, bipartite=0)
    g.add_nodes_from(target_nodes, bipartite=1)
    g.add_edges_from(edge_list)

    # todo: add graph attributes to both nodes
    giant_component, *_ = sorted(nx.connected_components(g), key=len, reverse=True)
    g = g.subgraph(giant_component)

    assert nx.is_connected(g)
    assert nx.is_bipartite(g)

    return g
