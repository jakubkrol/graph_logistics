import os 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def construct_file_path(parent, network_id, file_type):
    """
    Constructs a file path based on given parameters.
    """
    path_templates = {
        'network': "network_{network_id}.pkl",
        'result': "JK-network_{network_id}/optimisation/runResultinstances.xlsx",
        'distance': "selected_networks/network_{network_id}/0900hr-bikeDists.txt"
    }
    
    if file_type not in path_templates:
        raise ValueError("Invalid file type")
    
    return os.path.join(parent, path_templates[file_type].format(network_id=network_id))


def plot_results(network_id_1, network_id_2, coord_parent="network_zoo_radius_cluster_clean", 
                 results_parent="output_clust_size_clean", modes_to_plot=['bike', 'van', 'drone']):
    try:
        network_coord_path = construct_file_path(coord_parent, network_id_1, 'network')
        network_result_path = construct_file_path(results_parent, network_id_2, 'result')

        network_result = pd.read_excel(network_result_path, sheet_name='Routes')
        network_coord = pd.read_pickle(network_coord_path)

        if isinstance(network_coord, dict):
            network_coord = network_coord['Network']

        plot_network(network_coord, network_result, modes_to_plot)

    except Exception as e:
        print(f"An error occurred: {e}")


def plot_network(coordinates, routes, modes_to_plot=['bike', 'van', 'drone'], title="Route Visualization", xlabel="X Coordinate", ylabel="Y Coordinate"):
    """
    Plots routes on a scatter plot with each route in a different color.
    
    Parameters:
    coordinates (DataFrame): DataFrame containing 'x' and 'y' coordinates.
    routes (DataFrame): DataFrame containing route information.
    modes_to_plot (list): List of modes to plot.
    title (str): Title of the plot.
    xlabel (str): Label for the X-axis.
    ylabel (str): Label for the Y-axis.
    figsize (tuple): Size of the figure.
    marker (str): Marker style for the scatter plot.
    markersize (int): Size of the markers.
    linewidth (int): Width of the lines.
    """
    plot_base_network(coordinates)
    plot_routes(coordinates, routes, modes_to_plot)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show() 

def plot_base_network(coordinates, figsize=(10, 6), marker='o', markersize=20):
    plt.figure(figsize=figsize)
    plt.scatter(coordinates['x'], coordinates['y'], marker=marker, s=markersize)
    plt.grid(True)

def plot_routes(coordinates, routes, modes_to_plot, linewidth=2):
    line_styles = {'van': ':', 'bike': '-', 'drone': '--'}
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    def plot_line(start_node, end_node, style, color):
        x_values = [coordinates.loc[start_node, 'x'], coordinates.loc[end_node, 'x']]
        y_values = [coordinates.loc[start_node, 'y'], coordinates.loc[end_node, 'y']]
        plt.plot(x_values, y_values, style, color=color, linewidth=linewidth)
    
    color_index = 0
    for _, route in routes.iterrows():
        mode = route["Fulfilled By"].split(":")[1]
        if mode not in modes_to_plot:
            continue
        
        nodes_string = route['Route Stops'].strip("[]").split(",") if mode == 'bike' else route['Route Stops'].split(",")[1:-1]
        nodes = [int(node) for node in nodes_string]

        route_color = color_cycle[color_index % len(color_cycle)]
        route_style = line_styles[mode]

        depot = nodes[0] if mode == 'bike' else 0
        plot_line(depot, nodes[0], route_style, route_color)
        for j in range(len(nodes) - 1):
            plot_line(nodes[j], nodes[j + 1], route_style, route_color)
        plot_line(nodes[-1], depot, route_style, route_color)

        color_index += 1
