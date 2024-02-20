import os

import pandas as pd
import numpy as np

from scipy.spatial import distance_matrix as distance_matrix_f
from typing import Dict, Tuple


class NetworkAnalysis:
    @staticmethod
    def compute_clustering_coefficient(matrix):
        """Compute the clustering coefficient of a network represented by a binary adjacency matrix."""
        n = matrix.shape[0]
        degrees = np.sum(matrix, axis=1)
        den = np.sum(degrees * (degrees - 1))

        if den == 0:
            return 0

        triads = np.dot(matrix.T, matrix)
        triads = np.sum(np.multiply(matrix, triads))
        
        return triads / den

    def compute_network_clustering_coefficient(self, network, threshold=8):
        """Compute the clustering coefficient of a network based on a distance matrix with a specified threshold."""
        distance_mat = distance_matrix_f(network.values, network.values)
        distance_matrix_bin = np.where(distance_mat < threshold, 1, 0)
        np.fill_diagonal(distance_matrix_bin, 0)
        
        return self.compute_clustering_coefficient(distance_matrix_bin)

    @staticmethod
    def calculate_betweeness_centrality(network, node, threshold=0.05):
        """Calculates the betweenness centrality of a node within the given network."""
        n = network.shape[0]
        coordinates = network[['x', 'y']].values
        dist_no_node = distance_matrix_f(coordinates, coordinates)
        
        dist_i = np.sqrt(np.sum((coordinates - node) ** 2, axis=1))
        dist_matrix_i = dist_i[:, None] + dist_i  # creates a matrix of sums of distances
        ratio = (dist_matrix_i / dist_no_node - 1).flatten()  # flatten to make the shapes compatible

        # Mask to exclude the diagonal (self to self distances)
        mask = np.ones(dist_no_node.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        mask = mask.flatten()  # Flatten the mask to match the shape of ratio

        # Apply the mask to exclude diagonal and then apply the threshold
        shortest_path_count = np.sum((ratio < threshold) & mask)
        no_paths = n * (n - 1)  # Correct formula for number of paths excluding self paths
        return shortest_path_count / no_paths

class AnalysisOrchestrator:
    def __init__(self, output_to_zoo: Dict[str, str]):
        """
        Initialize the AnalysisOrchestrator instance.

        :param output_to_zoo: A dictionary mapping output folders to zoo identifiers.
        """
        self.output_to_zoo = output_to_zoo
        self.network_id = 0
        self.list_of_features = []
        self.all_selected_networks = {}

    def run_analysis(self):
        """
        Run analysis on directories that start with a specific prefix within the output folders.
        """
        prefix = "JK-network"  # Moved prefix out of the loop as it doesn't change per iteration
        for output_folder, zoo_id in self.output_to_zoo.items():
            # Use os.walk to navigate through the directory structure
            for root, dirs, files in os.walk(output_folder):
                # Filter directories starting with the prefix
                for dir_name in filter(lambda d: d.startswith(prefix), dirs):
                    folder_to_analyse = os.path.join(root, dir_name)
                    results = DataManagement(zoo_id).analyse_folder(folder_to_analyse)
                    if results is None:
                        continue
                    self.all_selected_networks[self.network_id] = folder_to_analyse
                    self.list_of_features.append(results)
                    self.network_id += 1

class DataManagement:
    def __init__(self, parent_folder: str):
        self.parent_folder = parent_folder

    def analyse_folder(self, folder_to_analyse: str) -> pd.Series:
        results, vehicle_counts = self._read_and_count_vehicles(folder_to_analyse)
        if results is None:
            return None
        distance_matrix = self._get_distance_matrix(folder_to_analyse)
        network_median_distance, network_mean_hospital = self._calculate_network_statistics(distance_matrix)
        network, clustering_coeff, centrality = self._get_network_and_analytics(folder_to_analyse)
        
        values = [
            network_median_distance, network_mean_hospital, clustering_coeff, centrality,
            *vehicle_counts, *map(bool, vehicle_counts)
        ]
        index = [
            'MedDist', 'MeanHosp', 'Cluster', 'Centre', 
            'NoVans', 'NoBikes', 'NoDrones', 
            'BoolVans', 'BoolBikes', 'BoolDrones'
        ]
        return pd.Series(values, index=index)

    def _read_and_count_vehicles(self, folder: str) -> Tuple[pd.DataFrame, Tuple[int, int, int]]:
        results_path = os.path.join(folder, "optimisation", "runResultinstances.xlsx")
        try:
            results = pd.read_excel(results_path)
        except:
            return None, None
        vehicle_ids = results['Vehicle ID']
        no_vans = vehicle_ids.str.contains(":van").sum()
        no_bikes = vehicle_ids.str.contains(":bike").sum()
        no_drones = vehicle_ids.str.contains(":drone").sum()
        return results, (no_vans, no_bikes, no_drones)

    def _get_distance_matrix(self, folder: str) -> pd.DataFrame:
        id_ = folder.split('_')[-1]
        distance_matrix_file = os.path.join(self.parent_folder, "selected_networks", f"network_{id_}", "0900hr-bikeDists.txt")
        return pd.read_csv(distance_matrix_file, sep="\t", index_col=0)

    def _calculate_network_statistics(self, distance_matrix: pd.DataFrame) -> Tuple[float, float]:
        network_median_distance = distance_matrix.median().median()
        network_mean_hospital = distance_matrix.iloc[0].iloc[1:].mean()
        return network_median_distance, network_mean_hospital

    def _get_network_and_analytics(self, folder: str) -> Tuple[pd.DataFrame, float, float]:
        id_ = folder.split('_')[-1]
        network_file = self._find_network_file(id_)
        network = pd.read_pickle(network_file)
        if isinstance(network, dict):
            network = network['Network']
        clustering_coeff = NetworkAnalysis().compute_network_clustering_coefficient(network)
        centrality = NetworkAnalysis.calculate_betweeness_centrality(network, (network.iloc[0]['x'], network.iloc[0]['y']))
        return network, clustering_coeff, centrality

    def _find_network_file(self, id_: str) -> str:
        network_folder = os.path.join(self.parent_folder)
        prefixed_files = [filename for filename in os.listdir(network_folder) if filename.startswith(f"network_{id_}")]
        if prefixed_files:
            return os.path.join(network_folder, prefixed_files[0])
        raise FileNotFoundError(f"No network file found for network_{id_}")