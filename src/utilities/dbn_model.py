from typing import List, Tuple
import re
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.models.DynamicBayesianNetwork import DynamicNode


class DBNModel:
    """
    A class to define, train, and perform inference with a Dynamic Bayesian Network (DBN).
    
    Dynamic Bayesian Networks represent temporal dependencies in multivariate time series data.
    This implementation supports model structure definition, parameter learning via Maximum 
    Likelihood Estimation, inference via log-likelihood computation, and various visualization methods.
    
    Attributes:
        model (DBN): The underlying pgmpy Dynamic Bayesian Network model
        intra_slice_edges (List[Tuple]): Edges connecting nodes within the same time slice
        inter_slice_edges (List[Tuple]): Edges connecting nodes between different time slices
    """

    def __init__(self, inter_slice_edges: List[Tuple], intra_slice_edges: List[Tuple]) -> None:
        """
        Initialize a Dynamic Bayesian Network (DBN) model with specified edge structure.
        
        Args:
            inter_slice_edges: List of edges connecting nodes between different time slices
                               Format: [(node1_t0, node2_t1), ...] where nodeN_tM is (name, time_slice)
            intra_slice_edges: List of edges connecting nodes within the same time slice
                               Format: [(node1_t0, node2_t0), ...] where nodeN_tM is (name, time_slice)
        
        Raises:
            Exception: If the model structure is invalid
        """
        # Initialize the DBN model with all edges
        self.model = DBN(intra_slice_edges + inter_slice_edges)

        # Check if the model structure is valid
        try: 
            self.model.check_model()
            print(
                f"Initialized and built the Dynamic Bayesian Network"
                f" with {len(self.model.nodes())} nodes"
                f" and {len(self.model.edges())} edges."
            )
        except Exception as e:
            print(f"Error while checking the DBN: {e}")
            raise
        
        # Store edge information for later use
        self.intra_slice_edges = intra_slice_edges
        self.inter_slice_edges = inter_slice_edges
    

    def train(self, df: pd.DataFrame) -> None:
        """
        Train the DBN using Maximum Likelihood Estimation (MLE) on the provided dataset.
        
        This method fits the conditional probability distributions (CPDs) of the network
        based on the observed frequencies in the training data.

        Args:
            df: DataFrame containing training data with columns corresponding to network variables.
                Each row represents an observation where columns are nodes at time t and t+1.
        
        Raises:
            Exception: If training fails due to data or model issues
        """
        try:
            print("Training the DBN model using Maximum Likelihood Estimation...")
            self.model.fit(df, estimator='MLE')
            print("DBN model training completed.")
        except Exception as e:
            print(f"Error during DBN training: {e}")
            raise
    
    def predict(self, data: pd.DataFrame, threshold: float) -> np.ndarray:
        """
        Generate binary predictions based on log-likelihood scores.
        
        Computes log-likelihood scores for each data point and classifies them as anomalous (1)
        if the score is below the threshold, or normal (0) otherwise.
        
        Args:
            data: DataFrame containing test data with columns corresponding to network variables
            threshold: Log-likelihood threshold for classification (below = anomalous)
            
        Returns:
            Array of binary predictions (1 = anomaly, 0 = normal)
        """
        predictions = []
        
        # Compute log-likelihood scores for each data point
        _, scores = self.compute_log_likelihood(data, show_progress=True)
        
        # Apply threshold to generate binary predictions
        for score in scores:
            predictions.append(1 if score < threshold else 0)
        
        return np.array(predictions)

    def print_cpds(self) -> None:
        """
        Print the Conditional Probability Distributions (CPDs) of the trained DBN model.
        
        This method displays the full conditional probability tables for each node
        in the network, showing how probabilities depend on parent node values.
        
        Raises:
            Exception: If CPDs cannot be retrieved or printed
        """
        try:
            for cpd in self.model.get_cpds():
                print(cpd)
            print("All CPDs successfully printed.")
        except Exception as e:
            print(f"Error while printing CPDs: {e}")

    def plot_structure(self) -> None:
        """
        Visualize the structure of the Dynamic Bayesian Network.
        
        Creates an interactive plot showing:
        - Nodes at time slice 0 in blue
        - Nodes at time slice 1 in light blue
        - Intra-slice edges (within same time slice) in matching colors
        - Inter-slice edges (between time slices) in red dashed lines
        
        The layout is optimized for readability with adjusted node positions.
        """
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add all edges with node names in the format "(name, time_slice)"
        for edge in self.model.edges():
            G.add_edge((edge[0][0], edge[0][1]), (edge[1][0], edge[1][1]))
        
        # Plot the graph
        plt.figure(figsize=(18, 8))
        pos = graphviz_layout(G, prog='dot')

        # Adjust positions for better visibility
        # This improves readability by preventing node overlap
        pos[('InverterTemp_RearRight_C', 0)] = (pos[('InverterTemp_RearRight_C', 0)][0] + 60, pos[('InverterTemp_RearRight_C', 0)][1])
        pos[('Inverter_Iq_Ref_RearRight_A', 0)] = (pos[('Inverter_Iq_Ref_RearRight_A', 0)][0], pos[('Inverter_Iq_Ref_RearRight_A', 0)][1] + 20)
        pos[('Inverter_Iq_Ref_RearLeft_A', 0)] = (pos[('Inverter_Iq_Ref_RearLeft_A', 0)][0], pos[('Inverter_Iq_Ref_RearLeft_A', 0)][1] + 20)
        pos[('MotorTemp_RearRight_C', 1)] = (pos[('MotorTemp_RearRight_C', 1)][0] - 150, pos[('MotorTemp_RearRight_C', 1)][1])
        pos[('MotorTemp_RearLeft_C', 1)] = (pos[('MotorTemp_RearLeft_C', 1)][0] + 150, pos[('MotorTemp_RearLeft_C', 1)][1])
        
        # Separate nodes by time slice
        t0_nodes = [node for node in G.nodes() if node[1] == 0]  # Time slice 0
        t1_nodes = [node for node in G.nodes() if node[1] == 1]  # Time slice 1
        
        # Draw nodes with different colors based on time slice
        node_size = 3000  # Increased node size for better visibility
        # Time slice 0 nodes - bright blue
        nx.draw_networkx_nodes(G, pos, 
                            nodelist=t0_nodes, 
                            node_color='royalblue',
                            node_size=node_size, 
                            alpha=0.8)
        # Time slice 1 nodes - lighter blue
        nx.draw_networkx_nodes(G, pos, 
                            nodelist=t1_nodes, 
                            node_color='lightsteelblue',
                            node_size=node_size, 
                            alpha=0.6)
        
        # Separate edges by time slice connectivity
        intra_t0_edges = [(u, v) for u, v in G.edges() if u[1] == 0 and v[1] == 0]  # Within t0
        intra_t1_edges = [(u, v) for u, v in G.edges() if u[1] == 1 and v[1] == 1]  # Within t1
        inter_edges = [(u, v) for u, v in G.edges() if u[1] == 0 and v[1] == 1]     # Between t0 and t1
        
        # Draw edges with different colors and styles
        # Intra-slice edges for time 0 - solid blue
        nx.draw_networkx_edges(G, pos, 
                            edgelist=intra_t0_edges, 
                            edge_color='royalblue',
                            arrows=True,
                            width=2)
        # Intra-slice edges for time 1 - solid light blue
        nx.draw_networkx_edges(G, pos, 
                            edgelist=intra_t1_edges, 
                            edge_color='lightsteelblue',
                            arrows=True,
                            width=1,
                            alpha=0.6)
        # Inter-slice edges - dashed red
        nx.draw_networkx_edges(G, pos, 
                            edgelist=inter_edges, 
                            edge_color='darkred',
                            arrows=True, 
                            style='dashed',
                            width=1)
        
        # Add labels to nodes
        labels = {node: f"({node[0]}, {node[1]})" for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold')
        
        plt.title("Dynamic Bayesian Network Structure", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def compute_log_likelihood(self, data: pd.DataFrame, show_progress: bool = True) -> Tuple[float, np.ndarray]:
        """
        Compute the log-likelihood scores for data given the trained DBN model.
        
        Calculates how likely the observed data is under the current model parameters.
        Lower log-likelihood values indicate less probable observations (potential anomalies).
        
        Args:
            data: DataFrame where each row contains values for all nodes at time t and t+1
            show_progress: Whether to display a progress bar during computation
            
        Returns:
            A tuple containing:
                - Total log-likelihood score (sum of all individual scores)
                - Array of log-likelihood scores for each data point
                
        Raises:
            Exception: If data format is incorrect or values not found in CPD states
        """
        # Get all the nodes in the network
        nodes = list(self.model.nodes())
        
        # We're only interested in calculating likelihood for nodes at time slice 1
        # since they depend on values in time slice 0
        nodes_t1 = [node for node in nodes if node[1] == 1]
        
        # Ensure data has the correct format
        if isinstance(data, pd.DataFrame):
            data_values = data.values
        else:
            raise Exception("Data must be in DataFrame format")
        
        # Initialize variables for log-likelihood calculation
        total_log_likelihood = 0
        log_likelihood_scores = []
        
        n_samples = len(data_values)
        
        # Create a mapping from node names to CPD objects for quick lookup
        node_to_cpd = {cpd.variable: cpd for cpd in self.model.get_cpds()}
        
        # Process each data point
        for i in tqdm(range(n_samples), desc="Computing log-likelihood", disable=not show_progress):
            row_log_likelihood = 0
            
            # Calculate log probability for each node at time t+1
            for node in nodes_t1:
                # Get the CPD for this node
                cpd = node_to_cpd.get(node)
                if cpd is None:
                    continue
                    
                # Get the current variable and its value
                current_var = node[0]
                current_val = data[(current_var, 1)].iloc[i]
                
                # Convert value to state number according to CPD
                # Each node has discrete states identified by indices in the CPD
                if node in cpd.state_names:
                    try:
                        current_no = cpd.state_names[node].index(current_val)
                    except (ValueError, IndexError):
                        raise ValueError(f"Value {current_val} not found in state_names for node {node}")
                    
                # Get the evidence variables (parents of the current node)
                evidence = self.model.get_parents(node)
                evidence_vals = {}
                
                # Collect values for all evidence variables
                for ev_node in evidence:
                    ev_var = ev_node[0]
                    ev_time = ev_node[1]

                    ev_val = data[(ev_var, ev_time)].iloc[i]
                    
                    # Convert to state number
                    if ev_node in cpd.state_names:
                        try:
                            ev_no = cpd.state_names[ev_node].index(ev_val)
                        except (ValueError, IndexError):
                            raise ValueError(f"Value {ev_val} not found in state_names for evidence {ev_node}")
                    
                    evidence_vals[ev_node] = ev_no
                
                # Get probability from CPD
                if evidence:
                    # Create index for evidence
                    evidence_index = []
                    for ev_node in cpd.variables[1:]:
                        if ev_node in evidence_vals:
                            evidence_index.append(evidence_vals[ev_node])
                        else:
                            # Skip if evidence is missing
                            break

                    # Extract probability from the CPD table using the evidence indices
                    prob = cpd.values[current_no][tuple(evidence_index)]
                else:
                    # No evidence, use marginal probability
                    prob = cpd.values[current_no]
                
                # Add log probability to sum (with a small epsilon to avoid log(0))
                epsilon = 1e-10  # Small constant to prevent log(0)
                prob = max(prob, epsilon)
                node_log_prob = np.log(prob)
                row_log_likelihood += node_log_prob
                
            # Add this row's log-likelihood to the total
            total_log_likelihood += row_log_likelihood
            log_likelihood_scores.append(row_log_likelihood)
        
        return total_log_likelihood, np.array(log_likelihood_scores)

    def plot_factors(self) -> None:
        """
        Plot the factor values for nodes at time slice 1 in the DBN.
        
        Factors represent the underlying probability values for each variable's states
        given its parents' states. This visualization helps understand the distribution
        of conditional probabilities and identify strong/weak dependencies.
        
        The plot includes reference lines at 0.8, 0.5, and 0.1 probability levels.
        """
        # Get nodes at time slice 1
        slice_1_nodes = [node for node in self.model.nodes() if node[1] == 1]
        
        # Create subplots - one for each node
        n_nodes = len(slice_1_nodes)
        fig, axes = plt.subplots(n_nodes, 1, figsize=(10, 3*n_nodes))
        
        # Ensure axes is always an array, even with a single subplot
        if n_nodes == 1:
            axes = [axes]
        
        # Plot factors for each node
        for i, node in enumerate(slice_1_nodes):
            # Get the CPD and convert to factor
            cpd = self.model.get_cpds(node)
            factor_values = cpd.to_factor().values
            
            # Get the parents of the node for subtitle
            evidence_vars = cpd.variables[1:]
            subtitle = f"Evidence: {', '.join(str(var) for var in evidence_vars)}"
            
            # Flatten factor values and sort for better visualization
            flat_values = factor_values.flatten()
            flat_values.sort()
            
            # Plot the sorted factor values
            axes[i].plot(flat_values, marker='o', linestyle='-')
            axes[i].set_title(f'Ordered Factors for {node[0]}\n{subtitle}', fontsize=10)
            axes[i].set_xlabel('Factors')
            axes[i].set_ylabel('Value')

            # Add reference horizontal lines at key probability levels
            axes[i].axhline(y=0.8, color='green', linestyle=':', alpha=0.7)  # High probability
            axes[i].axhline(y=0.5, color='black', linestyle=':', alpha=0.7)  # Medium probability
            axes[i].axhline(y=0.1, color='red', linestyle=':', alpha=0.7)    # Low probability

            # Add annotations for the reference lines
            axes[i].annotate('0.8', xy=(0, 0.8), xytext=(5, 1), 
                    color='green', 
                    xycoords='axes fraction',
                    textcoords='offset points',
                    fontsize='x-small')
            axes[i].annotate('0.5', xy=(0, 0.5), xytext=(5, 5), 
                    color='black', 
                    xycoords='axes fraction', 
                    textcoords='offset points',
                    fontsize='x-small')
            axes[i].annotate('0.1', xy=(0, 0.1), xytext=(5, 10), 
                    color='red', 
                    xycoords='axes fraction', 
                    textcoords='offset points',
                    fontsize='x-small')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_markov_blanket(self, target_node: DynamicNode) -> None:
        """
        Visualize the Markov blanket of a given node in the DBN.
        
        The Markov blanket of a node consists of its parents, children, and 
        other parents of its children. It represents the minimal set of nodes
        that shield the target node from the rest of the network.
        
        Args:
            target_node: The node for which to visualize the Markov blanket
                         in the form (name, time_slice)
        """
        # Get Markov blanket nodes for the target node
        markov_blanket = self.model.get_markov_blanket(target_node)
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add all edges with node names in the format "(name, time_slice)"
        for edge in self.model.edges():
            G.add_edge((edge[0][0], edge[0][1]), (edge[1][0], edge[1][1]))
        
        # Plot the graph
        plt.figure(figsize=(18, 8))
        pos = graphviz_layout(G, prog='dot')

        # Adjust positions for better visibility
        pos[('InverterTemp_RearRight_C', 0)] = (pos[('InverterTemp_RearRight_C', 0)][0] + 60, pos[('InverterTemp_RearRight_C', 0)][1])
        pos[('Inverter_Iq_Ref_RearRight_A', 0)] = (pos[('Inverter_Iq_Ref_RearRight_A', 0)][0], pos[('Inverter_Iq_Ref_RearRight_A', 0)][1] + 20)
        pos[('Inverter_Iq_Ref_RearLeft_A', 0)] = (pos[('Inverter_Iq_Ref_RearLeft_A', 0)][0], pos[('Inverter_Iq_Ref_RearLeft_A', 0)][1] + 20)
        pos[('MotorTemp_RearRight_C', 1)] = (pos[('MotorTemp_RearRight_C', 1)][0] - 150, pos[('MotorTemp_RearRight_C', 1)][1])
        pos[('MotorTemp_RearLeft_C', 1)] = (pos[('MotorTemp_RearLeft_C', 1)][0] + 150, pos[('MotorTemp_RearLeft_C', 1)][1])

        # Separate nodes by type
        blanket_nodes = [node for node in G.nodes() if node in markov_blanket and node != target_node]
        other_nodes = [node for node in G.nodes() if node not in blanket_nodes and node != target_node]
        
        # Draw nodes with different colors based on type
        node_size = 3000  # Increased node size for better visibility
        # Target node - blue
        nx.draw_networkx_nodes(G, pos, 
                               nodelist=[target_node], 
                               node_color='royalblue',
                               node_size=node_size, 
                               alpha=0.8)
        # Markov blanket nodes - coral
        nx.draw_networkx_nodes(G, pos, 
                               nodelist=blanket_nodes, 
                               node_color='lightcoral', 
                               node_size=node_size, 
                               alpha=0.8)
        # Other nodes - gray
        nx.draw_networkx_nodes(G, pos, 
                               nodelist=other_nodes, 
                               node_color='lightgray', 
                               node_size=node_size, 
                               alpha=0.5)
        
        # Separate edges by relevance to Markov blanket
        blanket_edges = [e for e in G.edges() if e[0] in markov_blanket + [target_node] and 
                                                e[1] in markov_blanket + [target_node]]
        other_edges = [e for e in G.edges() if e not in blanket_edges]

        # Draw edges with different colors and styles
        # Markov blanket edges - black
        nx.draw_networkx_edges(G, pos, 
                            edgelist=blanket_edges, 
                            edge_color='black',
                            arrows=True,
                            width=2)
        # Other edges - light gray
        nx.draw_networkx_edges(G, pos, 
                            edgelist=other_edges, 
                            edge_color='lightgray',
                            arrows=True,
                            width=1,
                            alpha=0.5)
        
        # Add labels to nodes
        labels = {node: f"({node[0]}, {node[1]})" for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold')
        
        plt.title(f"Markov Blanket of Node {target_node}", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def _split_local_independencies_str(self, independence_str: str) -> Tuple[List[DynamicNode], List[DynamicNode]]:
        """
        Parse a local independence statement string into component nodes.
        
        Takes a string representation of independence relations and extracts
        the d-separated nodes and observed nodes.
        
        Args:
            independence_str: String representation of independence in pgmpy format
                              e.g., "(X,0) ⟂ (Y,1),(Z,0) | (W,1)"
                              
        Returns:
            Tuple containing:
                - List of d-separated nodes (nodes independent of target given evidence)
                - List of observed nodes (evidence nodes)
        """
        # Remove spaces and split on the independence symbol
        independence_str = independence_str.replace(' ', '')
        independence_str = independence_str.split('⟂')[1]
        
        # Split into d-separated nodes and observed nodes
        dseparated_nodes_str, observed_nodes_str = independence_str.split('|')

        # Parse d-separated nodes
        dseparated_nodes = []
        matches = re.findall(r'\(([^()]+)\)', dseparated_nodes_str)
        for match in matches:
            name, time_slice = match.split(',')
            dseparated_nodes.append(DynamicNode(str(name), int(time_slice)))
        
        # Parse observed nodes
        observed_nodes = []
        matches = re.findall(r'\(([^()]+)\)', observed_nodes_str)
        for match in matches:
            name, time_slice = match.split(',')
            observed_nodes.append(DynamicNode(str(name), int(time_slice)))
        
        return dseparated_nodes, observed_nodes

    def visualize_local_independence(self, node: DynamicNode) -> None:
        """
        Visualize the local independencies for a given node in the DBN.
        
        Local independence shows which nodes are conditionally independent of
        the target node given some observed nodes (evidence). This helps understand
        the information flow in the network.
        
        Args:
            node: The target node in the form (name, time_slice)
        """
        # Get the target node
        current_node = node

        # Get the independence statement as a string
        independence_str = str(self.model.local_independencies(node))

        # Parse d-separated nodes (independent nodes) and observed nodes (evidence)
        dsep_nodes, evidence_nodes = self._split_local_independencies_str(independence_str)
        
        # Construct the DBN graph using networkx
        G = nx.DiGraph()
        for edge in self.model.edges():
            G.add_edge(edge[0], edge[1])
        
        # Compute active trails from current node given the evidence
        # Active trails represent paths through which information can flow
        active_trails = self.model.active_trail_nodes(
            current_node, observed=evidence_nodes, include_latents=True
        )
        
        # Get the set of nodes that are reached from current_node along active trails
        active_nodes_set = active_trails.get(current_node, set())
        
        # Determine which edges are part of these active trails
        active_trail_edges = []
        for u, v in G.edges():
            # We consider an edge active if both endpoints are reachable
            # or one endpoint is the current node
            if (u == current_node or v == current_node or 
                (u in active_nodes_set and v in active_nodes_set)):
                active_trail_edges.append((u, v))
        
        # Plot the graph
        plt.figure(figsize=(18, 8))
        pos = graphviz_layout(G, prog='dot')

        # Adjust positions for better visibility
        pos[('InverterTemp_RearRight_C', 0)] = (pos[('InverterTemp_RearRight_C', 0)][0] + 60, pos[('InverterTemp_RearRight_C', 0)][1])
        pos[('Inverter_Iq_Ref_RearRight_A', 0)] = (pos[('Inverter_Iq_Ref_RearRight_A', 0)][0], pos[('Inverter_Iq_Ref_RearRight_A', 0)][1] + 20)
        pos[('Inverter_Iq_Ref_RearLeft_A', 0)] = (pos[('Inverter_Iq_Ref_RearLeft_A', 0)][0], pos[('Inverter_Iq_Ref_RearLeft_A', 0)][1] + 20)
        pos[('MotorTemp_RearRight_C', 1)] = (pos[('MotorTemp_RearRight_C', 1)][0] - 150, pos[('MotorTemp_RearRight_C', 1)][1])
        pos[('MotorTemp_RearLeft_C', 1)] = (pos[('MotorTemp_RearLeft_C', 1)][0] + 150, pos[('MotorTemp_RearLeft_C', 1)][1])
        
        # Get nodes that are neither the target, evidence, nor d-separated
        other_nodes = [node for node in G.nodes() 
                      if node not in [current_node] + evidence_nodes + dsep_nodes]

        # Draw nodes with different colors based on type
        node_size = 3000
        # Current node - blue
        nx.draw_networkx_nodes(G, pos, 
                               nodelist=[current_node], 
                               node_color='royalblue',
                               node_size=node_size, 
                               alpha=0.8)
        # Evidence/observed nodes - gold
        nx.draw_networkx_nodes(G, pos, 
                               nodelist=evidence_nodes, 
                               node_color='gold', 
                               node_size=node_size, 
                               alpha=0.6)
        # D-separated nodes - purple
        nx.draw_networkx_nodes(G, pos, 
                               nodelist=dsep_nodes, 
                               node_color='plum', 
                               node_size=node_size, 
                               alpha=0.6)
        # Other nodes - gray
        nx.draw_networkx_nodes(G, pos, 
                               nodelist=other_nodes, 
                               node_color='lightgray', 
                               node_size=node_size, 
                               alpha=0.6)
        
        # Identify edges that are not part of active trails
        default_edges = [edge for edge in G.edges() if edge not in active_trail_edges]
        
        # Draw active trail edges in green, other edges in light gray
        nx.draw_networkx_edges(G, pos,
                            edgelist=active_trail_edges,
                            edge_color='limegreen',
                            arrows=True,
                            width=2)
        nx.draw_networkx_edges(G, pos,
                            edgelist=default_edges,
                            edge_color='lightgray',
                            arrows=True,
                            width=1,
                            alpha=0.6)
        
        # Add labels to nodes
        labels = {node: f"({node[0]}, {node[1]})" for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold')
        
        # Set title
        title_text = f"Local independencies for node {current_node}\n"
        plt.title(title_text, fontsize=16, fontweight='bold')

        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='royalblue', 
                      markersize=10, label='Current Node'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', 
                      markersize=10, label='Observed Nodes'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='plum', 
                      markersize=10, label='D-separated Nodes'),
            plt.Line2D([0], [0], color='limegreen', linewidth=2, label='Active Trails')
        ]
        plt.legend(handles=legend_elements, loc='upper left')

        plt.axis('off')
        plt.tight_layout()
        plt.show()