import copy
import math
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.interpolate import CubicSpline
from scipy.spatial import distance_matrix as distance_matrix_f

def calculate_betweeness_centrality(network, node, threshold=0.3):
    n = network.shape[0]
    shortest_path_count = 0
    for i in np.arange(n):
        for j in np.arange(n):
            if i == j:
                continue
            dist_no_node = np.sqrt((network.iloc[i]['x'] - network.iloc[j]['x']) ** 2 +
                                   (network.iloc[i]['y'] - network.iloc[j]['y']) ** 2)
            dist_i = np.sqrt((network.iloc[i]['x'] - node[0]) ** 2 + (network.iloc[i]['y'] - node[1]) ** 2)
            dist_j = np.sqrt((network.iloc[j]['x'] - node[0]) ** 2 + (network.iloc[j]['y'] - node[1]) ** 2)
            ratio = (dist_i + dist_j) / dist_no_node - 1
            if ratio < threshold:
                shortest_path_count += 1
    no_paths = n ** 2 - n # remove diagonal
    return shortest_path_count / no_paths

def calculate_similarity(reference_matrix, new_matrix):
    eigenvalues_ref = np.linalg.eig(reference_matrix)[0]
    eigenvalues_new = np.linalg.eig(new_matrix)[0]
    return np.linalg.norm(np.sort(eigenvalues_new) - np.sort(eigenvalues_ref)) / np.linalg.norm(eigenvalues_ref)

# Clustering coefficient functions
def compute_clustering_coeff(matrix):
    n = matrix.shape[0]
    den = 0
    for i in np.arange(n):
        k = 0
        for j in np.arange(n):
            k += matrix[i, j]
        den += k * (k - 1)
    if den == 0:
        return 0
    
    num = 0
    for i in np.arange(n):
        for j in np.arange(n):
            for k in np.arange(n):
                num += matrix[i, j] * matrix[j, k] * matrix[k, i]
    return num / den

def compute_clustering_coefficient(network):
    distance_mat = distance_matrix_f(network.values, network.values)
    threshold = 8
    distance_matrix_bin = copy.deepcopy(distance_mat)
    distance_matrix_bin[:] = np.where(distance_mat < threshold, 1, 0)
    dit_mat_bin_np = distance_matrix_bin
    np.fill_diagonal(dit_mat_bin_np, 0)
    coeff = compute_clustering_coeff(dit_mat_bin_np)
    return coeff

def compute_real_adjacency_matrix(network):
    distance_mat = distance_matrix_f(network.values, network.values)
    k = 0.1
    adjacency_matrix = copy.deepcopy(distance_mat)
    distance_matrix_adj = np.exp(- k* adjacency_matrix)
    np.fill_diagonal(distance_matrix_adj, 0)
    return distance_matrix_adj

def compute_degree_matrix(adjacency_matrix):
    return np.diag(adjacency_matrix.sum(axis=0))   

def compute_connectivity(network):
    adjacency_matrix = compute_real_adjacency_matrix(network)
    degree_matrix = compute_degree_matrix(adjacency_matrix)
    laplacian = degree_matrix - adjacency_matrix
    return np.sort(np.linalg.eig(laplacian)[0])[1]


def generate_network(r, std, N_surgeries = 100, box_radius=None):
    r_x = r / np.sqrt(2)
#     box_half_D = r + 3 * std
    box_half_D = box_radius if box_radius else 2 * max(r, std/2)
    box_half_x = box_half_D / np.sqrt(2)
    coord_surgery = []
    # Insert "hospital" in the centre of the network
    coord_surgery.append((0.0,  0.0))
    for i in range(N_surgeries - 1): # we subtract 1 because one of the surgeries is a hospital
        sample_choice = random.randint(1, 5)
        if sample_choice == 1:
            # Sample uniformly inside a box
            x_coord = random.uniform(-box_half_x, box_half_x)
            y_coord = random.uniform(-box_half_x, box_half_x)
        else:
            # Sample from Gaussian distribution and shift
            coord = np.random.multivariate_normal([0, 0], np.eye(2) * std, 1)
            x_coord = coord[0][0]
            y_coord = coord[0][1]

            if sample_choice == 2:
                # Shift to top left corner
                x_coord -= r_x
                y_coord += r_x

            if sample_choice == 3:
                # Shift to top right corner
                x_coord += r_x
                y_coord += r_x

            if sample_choice == 4:
                # Shift to bottom left corner
                x_coord -= r_x
                y_coord -= r_x

            if sample_choice == 5:
                # Shift to bottom right corner
                x_coord += r_x
                y_coord -= r_x
#         print(i, (x_coord, y_coord))
        coord_surgery.append((x_coord, y_coord))
    return pd.DataFrame(coord_surgery, columns=['x', 'y'])

class TrafficModel(object):
    
    def __init__(self, gm_data='edrone_traffic_data.csv', loc_data='locations_data.csv'):
        self.gm_data = pd.read_csv(gm_data)
        loc_data = pd.read_csv(loc_data, header=None)
        
        # read the list of southampton postcodes from somewhere FIXME
        self.dist_mat = pd.read_excel('distance_matrix.xlsx', index_col=0)
        self.southampton_postcodes = list(self.dist_mat.index)
        
        self.loc_data_filtered = loc_data[loc_data[1].isin(self.southampton_postcodes)]
        self.loc_data_filtered_id = list(self.loc_data_filtered[0])
    
    def fit(self, time):
        gm_data_filtered = self._get_filtered_gm_data(time)
        speed_mat = self._get_speed_model(gm_data_filtered)

        # convert to vectors
        dist_vector = self.dist_mat.values.flatten()
        speed_vector = speed_mat.values.flatten()
        self.dist_vector = dist_vector[np.isfinite(speed_mat.values.flatten())]
        self.speed_vector = speed_vector[np.isfinite(speed_mat.values.flatten())]
        
        self._fit_model()
    
    def _get_filtered_gm_data(self, time):
        day = 2
        gm_data_filtered = self.gm_data[self.gm_data[' day'] == day]
        gm_data_filtered = gm_data_filtered[gm_data_filtered[' time'] == time]
        gm_data_filtered = gm_data_filtered[gm_data_filtered[' origin_post'].isin(self.loc_data_filtered_id)]
        gm_data_filtered = gm_data_filtered[gm_data_filtered[' destination_post'].isin(self.loc_data_filtered_id)]
        return gm_data_filtered
    
    def _get_speed_model(self, gm_data_filtered):
    
        time_mat = pd.DataFrame(0, index=self.southampton_postcodes, columns=self.southampton_postcodes)

        for ind in time_mat.index:
            for col in time_mat.columns:
                origin_id = self.loc_data_filtered[self.loc_data_filtered[1] == ind][0]
                destination_id = self.loc_data_filtered[self.loc_data_filtered[1] == col][0]

                od_pair = gm_data_filtered[gm_data_filtered[' origin_post'] == origin_id.values[0]]
                od_pair = od_pair[od_pair[' destination_post'] == destination_id.values[0]]
                duration_traffic = np.nan if od_pair.shape[0] == 0 else od_pair[' duration_traffic'].iloc[0]

                time_mat.loc[ind, col] = duration_traffic

        speed_mat = self.dist_mat / time_mat
        return speed_mat
    
    def _fit_model(self):
        # fit cubic spline interpolant
        lengthscale = 50
        bins = np.arange(55)
        means = []
        stds = []
        for bin_ in bins:
            weights =  np.exp(-(self.dist_vector - bin_) ** 2 / lengthscale)
            mean, std = self.weighted_avg_and_std(self.speed_vector, weights)
            means.append(mean)
            stds.append(std)
        means = np.array(means)
        
        
        stds = np.array(stds)
        
        self.cs_means = CubicSpline(bins, means)
        self.cs_std =  CubicSpline(bins, stds)

        # fit extrapolant
        self.a, self.b = np.polyfit(np.log(bins[45:]), means[45:], 1)
        self.mean_std = np.mean(stds)
    
    @staticmethod
    def weighted_avg_and_std(values, weights):
        """
        Return the weighted average and standard deviation.

        values, weights -- NumPy ndarrays with the same shape.
        """
        average = np.average(values, weights=weights)
        # Fast and numerically precise:
        variance = np.average((values-average)**2, weights=weights)
        return (average, math.sqrt(variance))
        
    def predict(self, new_x):
        extrapolation_bins = np.arange(45, 75, 1)
        if new_x < 55:
            return self.cs_means(new_x), self.cs_std(new_x)
        else:
            return self.b + self.a * np.log(extrapolation_bins), self.mean_std
        
    def plot_model(self):
        bins = np.arange(55)
        plt.scatter(self.dist_vector, self.speed_vector)
        plt.plot(bins, self.cs_means(bins), 'red')
        plt.plot(bins, self.cs_means(bins) - self.cs_std(bins), 'red', linestyle='dashed')
        plt.plot(bins, self.cs_means(bins) + self.cs_std(bins), 'red', linestyle='dashed')

        extrapolation_bins = np.arange(45, 75, 1)
        plt.plot(extrapolation_bins, self.b + self.a * np.log(extrapolation_bins), 'red')
        plt.plot(extrapolation_bins, self.b + self.a * np.log(extrapolation_bins) - self.mean_std, 'red', 
                 linestyle='dashed')
        plt.plot(extrapolation_bins, self.b + self.a * np.log(extrapolation_bins) + self.mean_std, 'red', 
                 linestyle='dashed')

class TrafficModelKDE(object):
    
    def __init__(self, traffic_models_kde='kde_traffic_models.pkl', 
                 extrapolant_traffic_models='extrapolant_traffic_models.pkl'):
        
        with open(traffic_models_kde, 'rb') as handle:
            self.traffic_models_kde= pickle.load(handle)

        with open(extrapolant_traffic_models, 'rb') as handle:
            self.extrapolant_traffic_models= pickle.load(handle)
        
    def predict(self, new_x, time_of_day):
        if new_x < 51.5:
            traffic_model_kde = self.traffic_models_kde[time_of_day]
            # find closest bin
            
            index = round(new_x)
            traffic_model_kde = traffic_model_kde[index]
            travel_speed = traffic_model_kde.resample(1)[0]
            travel_speed = max(travel_speed, 0.000000001)
            travel_time = new_x / travel_speed 
            return travel_time 
        else:
            b = self.extrapolant_traffic_models[time_of_day]['b']
            a = self.extrapolant_traffic_models[time_of_day]['a']
            mean_std = self.extrapolant_traffic_models[time_of_day]['mean_std']
            mean = b + a * np.log(new_x)
            std = mean_std
            travel_speed = np.random.normal(mean, std)
            travel_speed = max(travel_speed, 0.000000001)
            travel_time = new_x / travel_speed 
            return travel_time         


class TimeMatrixGenerator(object):
    
    def __init__(self, distance_matrix, traffic_models, distance_matrix_ref, time_matrices_ref, 
                 times_of_day=np.arange(9, 18)):
        
        self.time_of_days = times_of_day  
        self.distance_matrix = distance_matrix
        self.traffic_models = traffic_models
        self.distance_matrix_ref = distance_matrix_ref
        self.time_matrices_ref = time_matrices_ref
        
        # vectorise sampling function
        self._sample_travel_time_vect = np.vectorize(self._sample_travel_time)
        
    def generate_time_matrices(self, n_matrices=100):
        # calculate reference similarity
        self.similarity_ref = self._calculate_spatial_similarity()
        # iterate through traffic models and generate time matrix for each
        self.time_matrices = []
        self.similarities = []
        for i, time_matrix_ref in enumerate(self.time_matrices_ref):
            time_of_day = self.time_of_days[i]
            time_matrix, similarity = self._generate_time_matrix(time_matrix_ref, 
                                                                 n_matrices, 
                                                                 time_of_day)
            self.time_matrices.append(time_matrix)
            self.similarities.append(similarity)
        return self.time_matrices
    
    def _calculate_spatial_similarity(self):
        return self._calculate_similarity(self.distance_matrix_ref, self.distance_matrix)
    
    @staticmethod
    def _calculate_similarity(reference_matrix, new_matrix):
        eigenvalues_ref = np.linalg.eig(reference_matrix)[0]
        eigenvalues_new = np.linalg.eig(new_matrix)[0]
        return np.linalg.norm(np.sort(eigenvalues_new) - np.sort(eigenvalues_ref)) / np.linalg.norm(eigenvalues_ref)
    
    def _generate_time_matrix(self, time_matrix_ref, n_matrices, time_of_day):
        self.best_time_matrix = np.random.rand(self.distance_matrix.shape[0], self.distance_matrix.shape[1]) * 1e6
        best_similarity = self._calculate_similarity(time_matrix_ref, self.best_time_matrix)
        best_similarity_diff = abs(best_similarity - self.similarity_ref)
        
        self.best_similarities = [] # DELETE AFTER DEBUGGING
        self.best_similarities.append(best_similarity) # DELETE AFTER DEBUGGING
        
        for i in np.arange(n_matrices):
            candidate_time_matrix = self._generate_single_time_matrix(time_of_day) 
            candidate_similarity = self._calculate_similarity(time_matrix_ref, candidate_time_matrix)
            candidate_similarity_diff = abs(candidate_similarity - self.similarity_ref)
            if candidate_similarity_diff < best_similarity_diff:
                self.best_time_matrix = candidate_time_matrix
                best_similarity = candidate_similarity
                best_similarity_diff = candidate_similarity_diff
            self.best_similarities.append(best_similarity) # DELETE AFTER DEBUGGING
            
        return self.best_time_matrix, best_similarity
    
    def _generate_single_time_matrix(self, time_of_day):
        return self._sample_travel_time_vect(self.distance_matrix, self.traffic_models, time_of_day)

    @staticmethod
    def _sample_travel_time(distance, traffic_models, time_of_day):
        return traffic_models.predict(distance, time_of_day)

# LEGACY METHOD FROM WHEN GAUSSIAN TRAFFIC MODEL WAS USED       
    # @staticmethod
    # def _sample_travel_time(distance, traffic_model):
    #     mean, std = traffic_model.predict(distance)
    #     travel_speed = np.random.normal(mean, std)
    #     travel_speed = max(travel_speed, 0.000000001)
    #     travel_time = distance / travel_speed 
    #     return travel_time 

    # def _sample_travel_time(self, distance):
    #     return self.traffic_models.predict(distance)


class AndyOFileConverter(object):
    
    def __init__(self, file_name, bike_speed=0.0052, drone_speed=0.028):
        with open(file_name, 'rb') as handle:
            network_matrices = pickle.load(handle)
        self.distance_matrix = network_matrices['Distance matrix']
        self.time_matrices = network_matrices['Time matrices'] # first 4 should be used 9, 10, 11, 12
        
        self.bike_speed = bike_speed
        self.drone_speed = drone_speed

        self.network_info = network_matrices['Network Info']
        
        # Create directory to save all the txt files
        self.new_directory = file_name[:-4]
        if not os.path.isdir(self.new_directory):
            os.mkdir(self.new_directory)
        
    def get_all_matrices(self):
        # get distance matrices
        self._save_distance_matrices()        
        # get time matrices
        self._save_time_matrices()
        # get summary file
        self._build_and_save_summary_file()

    def _save_distance_matrices(self):
        dist_matrix_size = self.distance_matrix.shape[0]
        distance_matrix_df = pd.DataFrame(self.distance_matrix, 
                                          index=np.arange(dist_matrix_size),
                                          columns=np.arange(dist_matrix_size))
        save_name = os.path.join(self.new_directory, '0900hr-bikeDists.txt')
        distance_matrix_df.to_csv(save_name, sep='\t', index=True)
        save_name = os.path.join(self.new_directory, '0900hr-droneDists.txt')
        distance_matrix_df.to_csv(save_name, sep='\t', index=True)
        save_name = os.path.join(self.new_directory, '0900hr-vanDists.txt')
        distance_matrix_df.to_csv(save_name, sep='\t', index=True)
        save_name = os.path.join(self.new_directory, '1000hr-bikeDists.txt')
        distance_matrix_df.to_csv(save_name, sep='\t', index=True)
        save_name = os.path.join(self.new_directory, '1000hr-droneDists.txt')
        distance_matrix_df.to_csv(save_name, sep='\t', index=True)
        save_name = os.path.join(self.new_directory, '1000hr-vanDists.txt')
        distance_matrix_df.to_csv(save_name, sep='\t', index=True)
        save_name = os.path.join(self.new_directory, '1100hr-bikeDists.txt')
        distance_matrix_df.to_csv(save_name, sep='\t', index=True)
        save_name = os.path.join(self.new_directory, '1100hr-droneDists.txt')
        distance_matrix_df.to_csv(save_name, sep='\t', index=True)
        save_name = os.path.join(self.new_directory, '1100hr-vanDists.txt')
        distance_matrix_df.to_csv(save_name, sep='\t', index=True)
        save_name = os.path.join(self.new_directory, '1200hr-bikeDists.txt')
        distance_matrix_df.to_csv(save_name, sep='\t', index=True)
        save_name = os.path.join(self.new_directory, '1200hr-droneDists.txt')
        distance_matrix_df.to_csv(save_name, sep='\t', index=True)
        save_name = os.path.join(self.new_directory, '1200hr-vanDists.txt')
        distance_matrix_df.to_csv(save_name, sep='\t', index=True)

    def _save_time_matrices(self):
        # Save van matrices
        dist_matrix_size = self.time_matrices[0].shape[0]
        distance_matrix_df = pd.DataFrame(self.time_matrices[0] / 60, 
                                          index=np.arange(dist_matrix_size),
                                          columns=np.arange(dist_matrix_size))
        save_name = os.path.join(self.new_directory, '0900hr-vanTimes.txt')
        distance_matrix_df.to_csv(save_name, sep='\t', index=True)

        distance_matrix_df = pd.DataFrame(self.time_matrices[1] / 60, 
                                          index=np.arange(dist_matrix_size),
                                          columns=np.arange(dist_matrix_size))
        save_name = os.path.join(self.new_directory, '1000hr-vanTimes.txt')
        distance_matrix_df.to_csv(save_name, sep='\t', index=True)


        distance_matrix_df = pd.DataFrame(self.time_matrices[2] / 60, 
                                          index=np.arange(dist_matrix_size),
                                          columns=np.arange(dist_matrix_size))
        save_name = os.path.join(self.new_directory, '1100hr-vanTimes.txt')
        distance_matrix_df.to_csv(save_name, sep='\t', index=True)


        distance_matrix_df = pd.DataFrame(self.time_matrices[3] / 60, 
                                          index=np.arange(dist_matrix_size),
                                          columns=np.arange(dist_matrix_size))
        save_name = os.path.join(self.new_directory, '1200hr-vanTimes.txt')
        distance_matrix_df.to_csv(save_name, sep='\t', index=True)
        
        # Save bike matrices
        dist_matrix_size = self.distance_matrix.shape[0]
        distance_matrix_df = pd.DataFrame(self.distance_matrix, 
                                          index=np.arange(dist_matrix_size),
                                          columns=np.arange(dist_matrix_size))
        bike_time_matrix = distance_matrix_df / self.bike_speed / 60

        save_name = os.path.join(self.new_directory, '0900hr-bikeTimes.txt')
        bike_time_matrix.to_csv(save_name, sep='\t', index=True)

        save_name = os.path.join(self.new_directory, '1000hr-bikeTimes.txt')
        bike_time_matrix.to_csv(save_name, sep='\t', index=True)

        save_name = os.path.join(self.new_directory, '1100hr-bikeTimes.txt')
        bike_time_matrix.to_csv(save_name, sep='\t', index=True)

        save_name = os.path.join(self.new_directory, '1200hr-bikeTimes.txt')
        bike_time_matrix.to_csv(save_name, sep='\t', index=True)

        # Save drone matrices
        drone_time_matrix = distance_matrix_df / self.drone_speed / 60
        save_name = os.path.join(self.new_directory, '0900hr-droneTimes.txt')
        drone_time_matrix.to_csv(save_name, sep='\t', index=True)

        save_name = os.path.join(self.new_directory, '1000hr-droneTimes.txt')
        drone_time_matrix.to_csv(save_name, sep='\t', index=True)

        save_name = os.path.join(self.new_directory, '1100hr-droneTimes.txt')
        drone_time_matrix.to_csv(save_name, sep='\t', index=True)

        save_name = os.path.join(self.new_directory, '1200hr-droneTimes.txt')
        drone_time_matrix.to_csv(save_name, sep='\t', index=True)

    def _build_and_save_summary_file(self):
        header = """bounding box min lat=50.78238005397546
bounding box min lon=-1.8312414853622556
bounding box max lat=51.00063389581439
bounding box max lon=-1.172123535130055
number of sites per case start=50
number of sites per case end=300
per case increase=50
time window earliest hour=9
drone avg speed (km/h)=100
drone site permission mean=0.0823
drone site permission std dev=0.0306
drone circuity mean=1.566
drone circuity std dev=0.1
c-Van suitable=0.02
*
"""
        input_size = 18525
        input_size_text = """Input Size:
{}
*
""".format(str(input_size))
        network_info = self.network_info
        network_info['Lat'] = network_info['x']
        network_info['Lon'] = network_info['y']
        network_info = network_info.drop(['x', 'y'], axis=1)
        network_info.index.name = "Site/Postcode"
        
        
        network_info['Van'] = 1
        network_info['Bike'] = 1
        network_info['Drone'] = 1
        network_info['C-Van Base'] = 0
        
        file_name = self.new_directory + '-summary.txt'
        _, file_name = os.path.split(file_name)
        text_file = open(os.path.join(self.new_directory, file_name), "w")
        n = text_file.write(header)
        n = text_file.write(input_size_text)
        n = text_file.write(network_info.to_string())
        text_file.close()
        
        file_name = self.new_directory + '-locationsOnly.csv'
        _, file_name = os.path.split(file_name)
        network_info.to_csv(os.path.join(self.new_directory, file_name), sep=',', index=True)