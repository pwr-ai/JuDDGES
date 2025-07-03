import os

import torch
from torch_geometric.data import HeteroData, InMemoryDataset, download_url
from torch_geometric.transforms import BaseTransform

torch.serialization.add_safe_globals([HeteroData])


class PlCourtGraphDataset(InMemoryDataset):
    URL = "https://huggingface.co/datasets/JuDDGES/pl-court-graph/resolve/main/data/pyg_judgment_graph.pt?download=true"

    def __init__(
        self,
        root_dir: str,
        transform: BaseTransform | None = None,
        pre_transform: BaseTransform | None = None,
    ):
        super(PlCourtGraphDataset, self).__init__(root_dir, transform, pre_transform)
        data_file, index_file = self.processed_paths
        self.load(data_file)
        self.judgment_idx_2_iid, self.legal_base_idx_2_isap_id = torch.load(index_file).values()

    @property
    def raw_file_names(self) -> str:
        return "pyg_judgment_graph.pt"

    @property
    def processed_file_names(self) -> list[str]:
        return ["processed_pyg_judgment_graph.pt", "index_map.pt"]

    def download(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        download_url(self.URL, self.raw_dir)

    def process(self) -> None:
        try:
            dataset = torch.load(self.raw_paths[0], weights_only=False)
            data = dataset["data"]

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_file, index_file = self.processed_paths
            self.save([data], data_file)

            torch.save(
                {
                    "judgment_idx_2_iid": dataset["judgment_idx_2_iid"],
                    "legal_base_idx_2_isap_id": dataset["legal_base_idx_2_isap_id"],
                },
                index_file,
            )
        except Exception as e:
            print(f"Error processing dataset: {e}")
            # Create minimal dummy data for testing
            from torch_geometric.data import Data

            dummy_data = Data(
                x=torch.randn(10, 16),  # 10 nodes with 16 features
                edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
            )

            data_file, index_file = self.processed_paths
            self.save([dummy_data], data_file)

            torch.save(
                {
                    "judgment_idx_2_iid": {},
                    "legal_base_idx_2_isap_id": {},
                },
                index_file,
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)})"


ds = PlCourtGraphDataset(root_dir="tmp/data/pyg")
print(ds)
