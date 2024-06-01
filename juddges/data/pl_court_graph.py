from pathlib import Path
from torch_geometric.data import HeteroData
from torch import Tensor
import polars as pl
import networkx as nx
import torch
from tqdm.auto import tqdm

FEATURES = ["_id", "signature", "date", "court_name", "department_name", "type", "isap_ids"]


def create_judgement_legal_base_pyg_graph(
    graph: nx.Graph,
    embeddings: dict[str, Tensor],
) -> tuple[HeteroData, dict[int, str]]:
    """Creates PytorchGeometric HeteroData object from networkx graph and judgment embeddings

    Args:
        graph (nx.Graph): graph created with create_judgement_legal_base_graph
        embeddings (dict[str, Tensor]): embeddings of judgments indexed by their original id

    Returns:
        tuple[HeteroData, dict[int, str]]: dataset and index to iid mapping
    """
    emb_dim = next(iter(embeddings.values())).size(0)
    num_judgements = len(nx.get_node_attributes(graph, "_id"))
    num_lb = len(nx.get_node_attributes(graph, "isap_id"))

    x_judgement = torch.zeros(num_judgements, emb_dim, dtype=torch.float)
    x_legal_base = torch.zeros(num_lb, 1, dtype=torch.float)

    index_2_idd = nx.get_node_attributes(graph, "_id")
    for index, iid in index_2_idd.items():
        x_judgement[index] = embeddings[iid]
    assert (x_judgement.sum(dim=1) != 0).all()

    edges = []
    for e in graph.edges:
        edges.append(e)
    edge_index = torch.tensor(edges).t()

    data = HeteroData(
        {
            "judgement": {"x": x_judgement},
            "legal_base": {"x": x_legal_base},
            ("judgement", "refers", "legal_base"): {"edge_index": edge_index},
        }
    )
    return data, index_2_idd


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
    df = (
        dataset.filter(pl.col("text_legal_bases").list.len() > 0)
        .with_columns(
            pl.col("text_legal_bases")
            .map_elements(lambda legal_bases: list(set(lb["isap_id"] for lb in legal_bases)))
            .alias("isap_ids")
        )
        .select(FEATURES)
        .collect()
    )

    if sample_size is not None:
        df = df.sample(sample_size)

    return df


def create_bipartite_graph(dataset: pl.DataFrame, verbose: bool) -> nx.Graph:
    isap_ids = dataset["isap_ids"].explode().unique()
    isap_id_2_index = {iid: index for index, iid in enumerate(isap_ids, start=len(dataset))}
    isap_attrs = {index: {"isap_id": iid} for iid, index in isap_id_2_index.items()}

    source_nodes = list(range(len(dataset)))
    target_nodes = list(isap_id_2_index.values())
    edge_list = []
    judgment_attrs = {}

    for index, doc in tqdm(
        enumerate(dataset.iter_rows(named=True)), total=len(dataset), disable=not verbose
    ):
        judgment_attrs[index] = {
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

    nx.set_node_attributes(g, judgment_attrs)
    nx.set_node_attributes(g, isap_attrs)

    # remove isolated small components
    giant_component, *_ = sorted(nx.connected_components(g), key=len, reverse=True)
    g = g.subgraph(giant_component)

    # make indexing contiguous
    g = nx.convert_node_labels_to_integers(g)

    assert nx.is_connected(g)
    assert nx.is_bipartite(g)

    return g
