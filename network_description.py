import os

import pandas as pd
import numpy as np

from scipy.spatial import distance_matrix as distance_matrix_f
from typing import Dict, Tuple


class NetworkAnalysis:

    @staticmethod
    def compute_clustering_coefficient(adjacency_matrix):
        """
        Compute the clustering coefficient of a network represented by a binary adjacency matrix.
        
        Parameters:
        - adjacency_matrix: A binary adjacency matrix representing the network.
        
        Returns:
        - Clustering coefficient as a float.
        """
        node_degrees = np.sum(adjacency_matrix, axis=1)
        denominator = np.sum(node_degrees * (node_degrees - 1))

        if denominator == 0:
            return 0  # Avoid division by zero if no connections exist

        triads_possible = np.dot(adjacency_matrix.T, adjacency_matrix)
        triads_closed = np.sum(np.multiply(adjacency_matrix, triads_possible))
        
        return triads_closed / denominator

    def compute_network_clustering_coefficient(self, network, threshold=8):
        """
        Compute the clustering coefficient of a network based on a distance matrix with a specified threshold.
        
        Parameters:
        - network_distances: Network coordinates.
        - threshold: Distance threshold to consider when calculating adjacency.
        
        Returns:
        - Network clustering coefficient as a float.
        """
        network_distances = distance_matrix_f(network.values, network.values)
        adjacency_matrix = np.where(network_distances < threshold, 1, 0)
        np.fill_diagonal(adjacency_matrix, 0)
        
        return self.compute_clustering_coefficient(adjacency_matrix)

    @staticmethod
    def calculate_betweeness_centrality(network, node, threshold=0.05):
        """
        Calculates the betweenness centrality of a node within a given network, based on distance matrices.

        Parameters:
        - network: A pandas DataFrame representing the network with 'x' and 'y' coordinates for each node.
        - node: The specific node coordinates (x, y) for which to calculate betweenness centrality.
        - threshold: The threshold ratio for considering a path to contribute to betweenness centrality.

        Returns:
        - The betweenness centrality value for the specified node as a float.
        """
        num_nodes = network.shape[0]
        coordinates = network[['x', 'y']].values
        dist_matrix_without_node = distance_matrix_f(coordinates, coordinates)
        
        dist_from_node = np.sqrt(np.sum((coordinates - node) ** 2, axis=1))
        sum_dist_matrix = dist_from_node[:, None] + dist_from_node  # Sums of distances to and from the node

        # Calculate the ratio of distances, adjusting for the node's presence
        distance_ratio = (sum_dist_matrix / dist_matrix_without_node - 1).flatten()

        # Prepare a mask to exclude self-comparisons
        exclusion_mask = np.ones(dist_matrix_without_node.shape, dtype=bool)
        np.fill_diagonal(exclusion_mask, 0)
        exclusion_mask = exclusion_mask.flatten()

        # Count paths shorter than the threshold, excluding self-paths
        relevant_path_count = np.sum((distance_ratio < threshold) & exclusion_mask)
        total_possible_paths = num_nodes * (num_nodes - 1)  # Total paths excluding self-paths
        return relevant_path_count / total_possible_paths
    
class AnalysisOrchestrator:
    def __init__(self, output_to_zoo: Dict[str, str]):
        """
        Initializes the AnalysisOrchestrator with mappings from output folders to zoo identifiers.
        
        Parameters:
        - output_to_zoo: A dictionary where keys are paths to output folders, and values are corresponding zoo identifiers.
        """
        self.output_to_zoo = output_to_zoo
        self.network_id_counter = 0
        self.features_collected = []
        self.selected_networks = {}

    def run_analysis(self):
        """
        Executes analysis on all directories within specified output folders that match a predefined prefix.
        
        It iterates through each output folder, identifies directories that start with the "JK-network" prefix,
        and performs analysis on those directories, collecting results and associating them with a network ID.
        """
        directory_prefix = "JK-network"
        for output_folder, zoo_id in self.output_to_zoo.items():
            for root, dirs, _ in os.walk(output_folder):
                for dir_name in filter(lambda d: d.startswith(directory_prefix), dirs):
                    analysis_path = os.path.join(root, dir_name)
                    analysis_results = DataManagement(zoo_id).analyse_folder(analysis_path)
                    if analysis_results is not None:
                        self.selected_networks[self.network_id_counter] = analysis_path
                        self.features_collected.append(analysis_results)
                        self.network_id_counter += 1

class DataManagement:
    def __init__(self, parent_folder: str):
        """
        Initializes the DataManagement instance with a parent folder path.
        
        Parameters:
        - parent_folder: The path to the parent folder where data files are stored.
        """
        self.parent_folder = parent_folder

    def analyse_folder(self, folder_to_analyse: str) -> pd.Series:
        results, vehicle_counts = self._read_and_count_vehicles(folder_to_analyse)
        if results is None:
            return None
        distance_matrix = self._get_distance_matrix(folder_to_analyse)
        median_distance, mean_hospital_distance = self._calculate_network_statistics(distance_matrix)
        network, clustering_coefficient, betweenness_centrality = self._get_network_and_analytics(folder_to_analyse)
        
        stats_values  = [
            median_distance, mean_hospital_distance, clustering_coefficient, betweenness_centrality,
            *vehicle_counts, *map(bool, vehicle_counts)
        ]
        stats_index = ['MedianDistance', 'MeanHospitalDistance', 'ClusteringCoefficient', 'BetweennessCentrality',
                        'Vans', 'Bikes', 'Drones', 'HasVans', 'HasBikes', 'HasDrones']
        return pd.Series(stats_values, index=stats_index)

    def _read_and_count_vehicles(self, folder: str) -> Tuple[pd.DataFrame, Tuple[int, int, int]]:
        """
        Reads vehicle data from an Excel file within the specified folder and counts the number of vans, bikes, and drones.
        
        Parameters:
        - folder: The path to the folder containing the vehicle data Excel file.
        
        Returns:
        - A tuple containing the DataFrame of results and a tuple of counts for vans, bikes, and drones.
        """
        results_path = os.path.join(folder, "optimisation", "runResultinstances.xlsx")
        try:
            results = pd.read_excel(results_path)
        except Exception:
            return None, None
        vehicle_ids = results['Vehicle ID']
        count_vans = vehicle_ids.str.contains(":van").sum()
        count_bikes = vehicle_ids.str.contains(":bike").sum()
        count_drones = vehicle_ids.str.contains(":drone").sum()
        return results, (count_vans, count_bikes, count_drones)

    def _get_distance_matrix(self, folder: str) -> pd.DataFrame:
        """
        Retrieves the distance matrix for the specified folder.
        
        Parameters:
        - folder: The path to the folder from which to retrieve the distance matrix.
        
        Returns:
        - A pandas DataFrame representing the distance matrix.
        """
        network_id = folder.split('_')[-1]
        distance_matrix_path = os.path.join(self.parent_folder, "selected_networks", f"network_{network_id}", "0900hr-bikeDists.txt")
        return pd.read_csv(distance_matrix_path, sep="\t", index_col=0)

    def _calculate_network_statistics(self, distance_matrix: pd.DataFrame) -> Tuple[float, float]:
        """
        Calculates median distance and mean distance to the nearest hospital in the network.
        
        Parameters:
        - distance_matrix: A pandas DataFrame representing the distance matrix of the network.
        
        Returns:
        - A tuple containing the network's median distance and mean distance to the nearest hospital.
        """
        median_distance = distance_matrix.median().median()
        mean_hospital_distance = distance_matrix.iloc[0].iloc[1:].mean()
        return median_distance, mean_hospital_distance

    def _get_network_and_analytics(self, folder: str) -> Tuple[pd.DataFrame, float, float]:
        """
        Retrieves network data and calculates clustering coefficient and betweenness centrality.
        
        Parameters:
        - folder: The path to the folder containing the network data.
        
        Returns:
        - A tuple containing the network DataFrame, clustering coefficient, and betweenness centrality.
        """
        network_id = folder.split('_')[-1]
        network_file = self._find_network_file(network_id)
        network = pd.read_pickle(network_file)
        if isinstance(network, dict):
            network = network['Network']
        clustering_coefficient = NetworkAnalysis().compute_network_clustering_coefficient(network)
        betweenness_centrality = NetworkAnalysis.calculate_betweeness_centrality(network, (network.iloc[0]['x'], network.iloc[0]['y']))
        return network, clustering_coefficient, betweenness_centrality

    def _find_network_file(self, network_id: str) -> str:
        """
        Finds the network file for the specified network ID within the parent folder.
        
        Parameters:
        - network_id: The ID of the network to find.
        
        Returns:
        - The path to the network file.
        
        Raises:
        - FileNotFoundError: If no network file is found for the given network ID.
        """
        network_folder = os.path.join(self.parent_folder)
        network_files = [filename for filename in os.listdir(network_folder) if filename.startswith(f"network_{network_id}")]
        if network_files:
            return os.path.join(network_folder, network_files[0])
        raise FileNotFoundError(f"No network file found for network_{network_id}")