import h3
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm


class H3NeighborDataset(Dataset):
    def __init__(self, data: pd.DataFrame, negative_sample_k_distance: int = 2):
        self.data = data
        self.data_torch = torch.Tensor(self.data.to_numpy())
        all_indices = set(data.index)

        self.inputs = []
        self.contexts = []
        self.input_h3 = []
        self.context_h3 = []

        self.positive_indexes = {}
        self.neighbours = {}

        for i, (h3_index, hex_data) in tqdm(
            enumerate(self.data.iterrows()), total=len(self.data)
        ):
            hex_neighbors_h3 = h3.k_ring(h3_index, 1)
            hex_neighbors_h3.remove(h3_index)
            available_neighbors_h3 = list(hex_neighbors_h3.intersection(all_indices))

            contexts_indexes = [
                self.data.index.get_loc(idx) for idx in available_neighbors_h3
            ]

            negative_excluded_h3 = h3.k_ring(h3_index, negative_sample_k_distance)
            negative_excluded_h3 = list(negative_excluded_h3.intersection(all_indices))
            positive_indexes = [
                self.data.index.get_loc(idx) for idx in negative_excluded_h3
            ]

            self.inputs.extend([i] * len(contexts_indexes))
            self.contexts.extend(contexts_indexes)
            try:
                self.positive_indexes[h3_index] = set(positive_indexes)
            except:
                print(h3_index)
                print(positive_indexes)

            self.input_h3.extend([h3_index] * len(available_neighbors_h3))
            self.context_h3.extend(available_neighbors_h3)
            self.neighbours[h3_index] = set(available_neighbors_h3)

        self.inputs = np.array(self.inputs)
        self.contexts = np.array(self.contexts)

        self.input_h3 = np.array(self.input_h3)
        self.context_h3 = np.array(self.context_h3)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input = self.data_torch[self.inputs[index]]
        context = self.data_torch[self.contexts[index]]
        input_h3 = self.input_h3[index]
        neg_index = self.get_random_negative_index(input_h3)
        negative = self.data_torch[neg_index]
        y_pos = 1.0
        y_neg = 0.0

        context_h3 = self.context_h3[index]
        negative_h3 = self.data.index[neg_index]
        return input, context, negative, y_pos, y_neg, input_h3, context_h3, negative_h3

    def get_random_negative_index(self, input_h3):
        excluded_indexes = self.positive_indexes[input_h3]
        negative = np.random.randint(0, len(self.data))
        while negative in excluded_indexes:
            negative = np.random.randint(0, len(self.data))
        return negative


class H3NeighborDataset2(Dataset):
    def __init__(self, data: pd.DataFrame, negative_sample_k_distance: int = 2):
        self.data = data
        self.data_torch = torch.Tensor(self.data.drop(columns="city").to_numpy())
        all_indices = set(data.index)

        self.inputs = []
        self.contexts = []
        self.input_h3 = []
        self.context_h3 = []

        self.positive_indexes = {}
        self.city = {}
        self.city_indexes = {}
        # self.neighbours = {}

        for i, h3_index in tqdm(enumerate(self.data.index), total=len(self.data)):
            city = self.data.loc[h3_index, "city"]
            self.city[h3_index] = city
            if city not in self.city_indexes.keys():
                self.city_indexes[city] = np.arange(len(self.data))[
                    self.data.city == city
                ]

            hex_neighbors_h3 = h3.k_ring(h3_index, 1)
            hex_neighbors_h3.remove(h3_index)
            hex_neighbors_h3 = list(hex_neighbors_h3.intersection(all_indices))

            contexts_indexes = [
                self.data.index.get_loc(idx) for idx in hex_neighbors_h3
            ]

            self.inputs.extend([i] * len(contexts_indexes))
            self.contexts.extend(contexts_indexes)

            positive_indexes_h3 = h3.k_ring(h3_index, negative_sample_k_distance)
            positive_indexes_h3 = list(positive_indexes_h3.intersection(all_indices))
            positive_indexes = [
                self.data.index.get_loc(idx) for idx in positive_indexes_h3
            ]

            self.positive_indexes[h3_index] = set(positive_indexes)

            self.input_h3.extend([h3_index] * len(hex_neighbors_h3))
            self.context_h3.extend(hex_neighbors_h3)

        self.inputs = np.array(self.inputs)
        self.contexts = np.array(self.contexts)

        self.input_h3 = np.array(self.input_h3)
        self.context_h3 = np.array(self.context_h3)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input = self.data_torch[self.inputs[index]]
        context = self.data_torch[self.contexts[index]]
        input_h3 = self.input_h3[index]
        neg_index = self.get_random_negative_index(input_h3)
        negative = self.data_torch[neg_index]
        y_pos = 1.0
        y_neg = 0.0

        context_h3 = self.context_h3[index]
        negative_h3 = self.data.index[neg_index]
        return input, context, negative, y_pos, y_neg, input_h3, context_h3, negative_h3

    def get_random_negative_index(self, input_h3):
        city_indexes = self.city_indexes[self.city[input_h3]]
        excluded_indexes = self.positive_indexes[input_h3]
        negative = city_indexes[np.random.randint(0, len(city_indexes))]
        while negative in excluded_indexes:
            negative = city_indexes[np.random.randint(0, len(city_indexes))]
        return negative


class H3NeighborTrainsetBRW(Dataset):
    def __init__(
        self,
        data_brw_features: pd.DataFrame,
        data_labels,
        data_h2v_features: pd.DataFrame,
    ):
        self.h2v = torch.Tensor(
            pd.merge(data_brw_features, data_h2v_features, on="h3").to_numpy()
        )

        all_indices = set(data_h2v_features.index)
        h2v_neighbours = []
        for h3_index in data_brw_features.index:
            hex_neighbors_h3 = h3.k_ring(h3_index, 1)
            hex_neighbors_h3.remove(h3_index)
            available_neighbors_h3 = list(hex_neighbors_h3.intersection(all_indices))
            h2v_neighbours.append(
                data_h2v_features.loc[available_neighbors_h3].sum()
                / len(available_neighbors_h3)
            )
        self.h2v_neighbours = torch.Tensor(h2v_neighbours)

        self.X = torch.cat((self.h2v, self.h2v_neighbours), 1)

        self.label = torch.Tensor(data_labels.to_numpy())

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        label = self.label[index]
        return x, label
