import os
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional
import h5py


class MovingMNIST(Dataset):
    def __init__(self, data_paths: List[str], preload: bool = False):
        super().__init__()
        self.data_paths: List[str] = data_paths
        self.preload: bool = preload

        self._index: List[Tuple[int, int]] = [] 
        self._tensor_cache: Dict[int, torch.Tensor] = {}
        self._h5_files: Dict[int, "h5py.File"] = {}

        for file_idx, path in enumerate(self.data_paths):
            with h5py.File(path, "r", libver="latest") as f:
                if "frames" not in f:
                    raise KeyError(f"Missing 'frames' dataset in HDF5 file: {path}")
                num_seq = f["frames"].shape[0]
            self._index.extend([(file_idx, s) for s in range(num_seq)])

    def __len__(self) -> int:
        return len(self._index)

    def _get_h5_file(self, file_idx: int):
        f = self._h5_files.get(file_idx)
        if f is None:
            path = self.data_paths[file_idx]
            f = h5py.File(path, "r", libver="latest", swmr=False)
            self._h5_files[file_idx] = f
        return f

    def _get_h5_sequence(self, file_idx: int, seq_idx: int) -> torch.Tensor:
        f = self._get_h5_file(file_idx)
        ds = f["frames"]  # (N, T, 3, H, W)
        arr = ds[seq_idx]  # numpy array view/copy
        tensor = torch.from_numpy(arr)
        if tensor.dtype == torch.uint8:
            tensor = tensor.float().div_(255.0)
        return tensor

    def _get_pt_frames_tensor(self, file_idx: int) -> torch.Tensor:
        frames = self._tensor_cache.get(file_idx)
        if frames is None:
            frames = torch.load(self.data_paths[file_idx], map_location="cpu")["frames"]
            self._tensor_cache[file_idx] = frames 
        return frames

    def __getitem__(self, idx: int) -> torch.Tensor:
        file_idx, seq_idx = self._index[idx]
        path = self.data_paths[file_idx]
        tensor = self._get_h5_sequence(file_idx, seq_idx)
        return tensor

    def __del__(self):
        for f in self._h5_files.values():
            try:
                f.close()
            except Exception:
                pass
        self._h5_files.clear()