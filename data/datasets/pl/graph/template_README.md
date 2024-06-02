---
language: {{language}}
pretty_name: {{pretty_name}}
size_categories: {{size_categories}}
source_datasets: {{source_datasets}}
viewer: {{viewer}}
tags: {{tags}}
---

# Polish Court Judgements Graph

## Dataset description
We introduce a graph dataset of Polish Court Judgements. This dataset is primarily based on the [`JuDDGES/pl-court-raw`](https://huggingface.co/datasets/JuDDGES/pl-court-raw). The dataset consists of nodes representing either judgments or legal bases, and edges connecting judgments to the legal bases they refer to. Consequently, the resulting graph is bipartite.

We provide the dataset in both `JSON` and `PyTorch Geometric` formats. The `JSON` format contains the following attributes (see [`JuDDGES/pl-court-raw`](https://huggingface.co/datasets/JuDDGES/pl-court-raw) for details). The `PyTorch Geometric` format includes embeddings of the judgment content, obtained with `{{embedding_method}}` for judgment nodes, and tensor of zeros for legal-base nodes.

## Dataset statistics

| feature                     | value                |
|-----------------------------|----------------------|
| #nodes                      | {{num_nodes}}        |
| #edges                      | {{num_edges}}        |
| #nodes of type `judgement`  | {{num_src_nodes}}    |
| #nodes of type `legal-base` | {{num_target_nodes}} |
| avg. degree                 | {{avg_degree|round(2)}}       |

## Load `JSON`
Graph the `JSON` format is saved in node-link format, and can be readily loaded with `networkx` library:

```python
import json
import network as nx

with open("<path_to_json_file>") as file:
    g_data = json.load(file)

g = nx.node_link_graph(g_data)
```

## Load `Pytorch Geometric`

In order to load graph as pytorch geometric, one can leverage the following code snippet
```python
# TBD
```