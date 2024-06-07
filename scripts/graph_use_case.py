import torch
import os
from torch_geometric.data import InMemoryDataset, download_url


class PlCourtGraphDataset(InMemoryDataset):
    URL = (
        "https://huggingface.co/datasets/JuDDGES/pl-court-graph/resolve/main/"
        "data/pyg_judgement_graph.pt?download=true"
    )

    def __init__(self, root_dir: str, transform=None, pre_transform=None):
        super(PlCourtGraphDataset, self).__init__(root_dir, transform, pre_transform)
        data_file, index_file = self.processed_paths
        self.load(data_file)
        self.judgement_idx_2_iid, self.legal_base_idx_2_isap_id = torch.load(index_file).values()

    @property
    def raw_file_names(self) -> str:
        return "pyg_judgement_graph.pt"

    @property
    def processed_file_names(self) -> list[str]:
        return ["processed_pyg_judgement_graph.pt", "index_map.pt"]

    def download(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        download_url(self.URL + self.raw_file_names, self.raw_dir)

    def process(self) -> None:
        dataset = torch.load(self.raw_paths[0])
        data = dataset["data"]

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        data_file, index_file = self.processed_paths
        self.save([data], data_file)

        torch.save(
            {
                "judgement_idx_2_iid": dataset["judgement_idx_2_iid"],
                "legal_base_idx_2_isap_id": dataset["legal_base_idx_2_isap_id"],
            },
            index_file,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)})"


ds = PlCourtGraphDataset(root_dir="data/datasets/pyg")
print(ds)
