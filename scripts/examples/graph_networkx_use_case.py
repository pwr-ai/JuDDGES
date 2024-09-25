import json

import networkx as nx
from huggingface_hub import hf_hub_download

DATA_DIR = "/tmp/data"
JSON_FILE = "data/judgment_graph.json"
hf_hub_download(
    repo_id="JuDDGES/pl-court-graph", repo_type="dataset", filename=JSON_FILE, local_dir=DATA_DIR
)

with open(f"{DATA_DIR}/{JSON_FILE}") as file:
    g_data = json.load(file)

g = nx.node_link_graph(g_data)
