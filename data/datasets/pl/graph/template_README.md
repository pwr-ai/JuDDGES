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
We introduce a graph dataset of Polish Court Judgements. This dataset is primarily based on the [`JuDDGES/pl-court-raw`](https://huggingface.co/datasets/JuDDGES/pl-court-raw). The dataset consists of nodes representing either judgments or legal bases, and edges connecting judgments to the legal bases they refer to. Consequently, the resulting graph is bipartite. We provide the dataset in both `JSON` and `PyG` formats, each has different purpose. While structurally graphs in these formats are the same, their attributes differ. 

The `JSON` format is intended for analysis and contains most of the attributes available in [`JuDDGES/pl-court-raw`](https://huggingface.co/datasets/JuDDGES/pl-court-raw). We excluded some less-useful attributes and text content, which can be easily retrieved from the raw dataset and added to the graph as needed.

The `PyG` format is designed for machine learning applications, such as link prediction on graphs, and is fully compatible with the [`Pytorch Geometric`](https://github.com/pyg-team/pytorch_geometric) framework. 

In the following sections, we provide a more detailed explanation and use case examples for each format.

## Dataset statistics

| feature                     | value                |
|-----------------------------|----------------------|
| #nodes                      | {{num_nodes}}        |
| #edges                      | {{num_edges}}        |
| #nodes of type `judgement`  | {{num_src_nodes}}    |
| #nodes of type `legal-base` | {{num_target_nodes}} |
| avg. degree                 | {{avg_degree|round(2)}}       |

## `JSON` format
The `JSON` format contains graph node types differentiated by `node_type` attrbute. Each `node_type` has its additional corresponding attributes (see [`JuDDGES/pl-court-raw`](https://huggingface.co/datasets/JuDDGES/pl-court-raw) for detailed description of each attribute):

| node_type    | attributes                                                                                                          |
|--------------|---------------------------------------------------------------------------------------------------------------------|
| `judgement`  | `_id`, `signature`, `date`, `court_name`, `department_name`, `type`, `judges`, `chairman`, `publisher`, `recorder`  |
| `legal_base` | `isap_id`, `title`                                                                                                  |

 
### Loading
Graph the `JSON` format is saved in node-link format, and can be readily loaded with `networkx` library:

```python
import json
import network as nx

with open("<path_to_json_file>") as file:
    g_data = json.load(file)

g = nx.node_link_graph(g_data)
```

### Example usage
```python
# TBD
```

## `PyG` format
The `PyTorch Geometric` format includes embeddings of the judgment content, obtained with `{{embedding_method}}` for judgment nodes, and one-hot-vector identifiers for legal-base nodes (note that for efficiency one can substitute it with random noise identifiers, like in [(Abboud et al., 2021)](https://arxiv.org/abs/2010.01179)).

### Loading
In order to load graph as pytorch geometric, one can leverage the following code snippet
```python
# TBD
```

### Example usage
```python
# TBD
```