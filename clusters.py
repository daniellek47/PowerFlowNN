import numpy as np
import pandas as pd
import ast
from sklearn.preprocessing import StandardScaler
from train_clusters import train_and_evaluate_nn
from train_clusters_improved import train_simple_nn
import csv
import itertools
from visualize import simple_output_visualization, inspect_data_columns, find_value_ranges
class Node:
    def __init__(self, num, indexes, names, loads):
        self.num = num
        self.indexes = indexes
        self.names = names
        self.loads = loads
        
    def __str__(self):
        return f"{self.num}:\nindexes:{self.indexes}\nnames:{self.names}\nloads:{self.loads}"

def parse_cluster_info(textfile, load_indices):
    """
    Parse load definitions with detailed connection information
    
    Returns a dictionary with detailed load characteristics
    """
    clusters = {}
    for line in textfile.split('\n'):
        if line.startswith('#'):
            continue
        if line == '':
            return clusters
        else:
            parts = line.split()
            if(parts[0][0] in clusters):
                clusters[parts[0][0]][0].append(int(parts[3]))
            else:
                clusters[parts[0][0]] = [[int(parts[3])], [], []]
            if(int(parts[3]) in load_indices):
                idx = load_indices.index(int(parts[3]))
                clusters[parts[0][0]][1].append(idx)
                clusters[parts[0][0]][2].append(int(parts[3]))

        
def map_load_indices_to_node_names(load_indices, node_names, load_info):
    """
    Map load indices to their corresponding node names, phases, and dataset indexes
    
    Parameters:
    - load_indices: Indices of nodes with loads
    - node_names: List of node names
    - load_info: Parsed load information
    
    Returns:
    - Dictionary with detailed mapping information
    """
    load_node_mapping = {}
    
    # First, collect all unique base nodes from load_info
    all_load_nodes = set(load_info.keys())
    
    # Iterate through all nodes with loads
    for base_node in all_load_nodes:
        # Create flexible matching patterns
        matching_patterns = [
            f"{base_node}.",
            f"{base_node}r.",
            f"{base_node}s.",
            f"{base_node}_open.",
            f"{base_node}.1",
            f"{base_node}.2",
            f"{base_node}.3",
            f"{base_node}.1.2",
            f"{base_node}.2.3",
            f"{base_node}.3.1"
        ]
        
        # Find all nodes with matching patterns
        matching_nodes = [
            node for node in node_names 
            if any(pattern in node for pattern in matching_patterns)
        ]
        
        # Find corresponding dataset indexes for amplitudes
        amplitude_indexes = [
            i for i, node in enumerate(node_names[:278]) 
            if any(pattern in node for pattern in matching_patterns)
        ]
        
        # Find corresponding dataset indexes for phases
        phase_indexes = [
            i for i, node in enumerate(node_names[278:], start=278) 
            if any(pattern in node for pattern in matching_patterns)
        ]
        
        # Fallback for nodes with no direct matches
        if not amplitude_indexes and not phase_indexes:
            amplitude_indexes = [
                i for i, node in enumerate(node_names[:278]) 
                if base_node in node
            ]
            phase_indexes = [
                i for i, node in enumerate(node_names[278:], start=278) 
                if base_node in node
            ]
        
        # Calculate total load for the node
        total_load = sum(load['total_load'] for load in load_info.get(base_node, []))
        
        # Ensure mapping 
        load_node_mapping[base_node] = {
            'matching_nodes': matching_nodes,
            'num_phases': len(set(node.split('.')[1] for node in matching_nodes)) if matching_nodes else 1,
            'amplitude_indexes': amplitude_indexes,
            'phase_indexes': phase_indexes,
            'total_load': total_load
        }
    
    return load_node_mapping

def filter_output_columns(df, column_indices, add_offset=None):
    """
    Filter a DataFrame to keep only specific columns by their integer positions.
    Optionally include additional columns at position index+offset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame
    column_indices : list
        List of integer column indices to keep
    add_offset : int, optional
        If provided, also include columns at positions index+offset
        
    Returns:
    --------
    pandas.DataFrame
        A new DataFrame containing only the selected columns
    """
    # Create a copy of the original indices
    all_indices = column_indices.copy()
    
    # If offset is provided, add the offset columns
    if add_offset is not None:
        offset_indices = [idx + add_offset for idx in column_indices]
        all_indices.extend(offset_indices)
    
    # Select the columns using iloc
    filtered_df = df.iloc[:, all_indices]
    return filtered_df
def filter_input_columns(df, column_indices):
    """
    Filter a DataFrame to keep only specific columns by their integer positions.
    Optionally include additional columns at position index+offset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame
    column_indices : list
        List of integer column indices to keep
    add_offset : int, optional
        If provided, also include columns at positions index+offset
        
    Returns:
    --------
    pandas.DataFrame
        A new DataFrame containing only the selected columns
    """
    # Create a copy of the original indices
    all_indices = []
    for indice in column_indices:
        all_indices.append(2*int(indice))
        all_indices.append(2*int(indice)+1)

    
    # Select the columns using iloc
    filtered_df = df.iloc[:, all_indices]
    return filtered_df

def get_dataset(clusters, data_in, data_out):
    cluster_datasets = {}
    
    for cluster in clusters:
        # Handle output data indexes
        out_indexes = clusters[cluster][0].copy()  # Create a copy to avoid modifying the original
        
        # Add index+278 for each index in the output indexes
        for index in clusters[cluster][0]:
            out_indexes.append(index + 278)
            
        # Handle input data indexes
        in_indexes = []
        for index in clusters[cluster][1]:
            in_indexes.append(2 * index)
            in_indexes.append(2 * index + 1)
 
        # Filter the dataframes using the collected indexes
        cluster_in_data = data_in.iloc[:, in_indexes] 
        cluster_out_data = data_out.iloc[:, out_indexes]
        
        # Store the datasets in the result dictionary
        cluster_datasets[cluster] = (cluster_in_data, cluster_out_data)
        
    return cluster_datasets



def compute_node_power(voltages_df, admittance_row, target_node_index, node_names, neighbors):
    """
    Compute complex power for a specific node using admittance matrix
    
    Parameters:
    -----------
    voltages_df : pandas.DataFrame
        DataFrame containing voltage magnitudes and phases
    admittance_matrix_df : pandas.DataFrame
        DataFrame with admittance values between nodes
    target_node : str
        Node for which power is being calculated
    
    Returns:
    --------
    dict: Containing active power (P) and reactive power (Q)
    """
    # Prepare arrays for P and Q
    P = np.zeros(len(voltages_df))
    Q = np.zeros(len(voltages_df))

    # Detailed calculation logging
    detailed_log = []

    # Iterate through each row
    for index, row in voltages_df.iterrows():
        # Extract target node's voltage
        V_i_mag = row[target_node_index]
        V_i_phase = row[target_node_index+278]
        V_i_complex = V_i_mag * np.exp(1j * V_i_phase)
        
        # Compute power injection
        total_power = 0
        row_log = []

        # Iterate through non-zero admittance values
        for col_name, Y_ij in admittance_row.items():
            # Skip empty, zero, or self-admittance values
            if col_name not in neighbors:
                continue
            
            try:
                # Find the column index for this node
                col_index = node_names.index(col_name)
                
                # Extract connected node's voltage
                V_j_mag = row[col_index]
                V_j_phase = row[col_index+278]
                V_j_complex = V_j_mag * np.exp(1j * V_j_phase)
                # Ensure Y_ij is a complex number
                Y_ij_complex = complex(Y_ij)
                # Compute power injection
                #S_ij = Y_ij * V_j_complex * np.conj(V_i_complex)
                Y_ij_V_j = Y_ij * V_j_complex
                S_ij = Y_ij_V_j*np.conj(V_i_complex)

                # Log detailed calculation
                row_log.append({
                    'connected_node': col_name,
                    'V_i': V_i_complex,
                    'V_j': V_j_complex,
                    'Y_ij': Y_ij,
                    'Y_ij_V_j': Y_ij_V_j,
                    'S_ij': S_ij
                })

                total_power += S_ij
                
            except (ValueError, KeyError, IndexError) as e:
                # Skip if voltage for this node not found
                print(f"Error processing {col_name}: {e}")
                continue
        # Store real and imaginary parts
        P[index] = total_power.real
        Q[index] = total_power.imag
        
        # Store detailed log for the first few rows
        if index < 5:
            detailed_log.append(row_log)

    # Print detailed log for the first few rows
    for i, row_log in enumerate(detailed_log):
        print(f"\nDetailed calculation for row {i}:")
        for entry in row_log:
            print(f"Connected Node: {entry['connected_node']}")
            print(f"V_i: {entry['V_i']}")
            print(f"V_j: {entry['V_j']}")
            print(f"Y_ij: {entry['Y_ij']}")
            print(f"Y_ij * V_j: {entry['Y_ij_V_j']}")
            print(f"S_ij: {entry['S_ij']}")
            print("---")

    # Create DataFrame with P and Q columns
    return pd.DataFrame({'P': P, 'Q': Q})
def parse_complex(x):
    if isinstance(x, str):
        try:
            # Remove parentheses and convert
            x = x.strip('()')
            return complex(x.replace('j', 'j'))
        except ValueError:
            # If parsing fails, return 0j or handle as needed
            return 0j
    return x

def main():
    # Usage
    # output indexes indicate that there is a load connected to this phase.
    load_indices = [9, 12, 17, 18, 19, 14, 24, 35, 34, 23, 37, 38, 39, 43,
    44, 48, 56, 63, 68, 61, 72, 62, 30, 81, 82, 83, 84, 85,
    88, 92, 96, 100, 101, 102, 105, 106, 112, 115, 121, 128, 130, 134,
    131, 140, 141, 145, 152, 156, 163, 164, 165, 166, 170, 171, 173, 181,
    185, 190, 196, 193, 197, 176, 199, 201, 205, 209, 213, 215, 217, 218,
    222, 226, 236, 240, 241, 242, 246, 247, 252, 253, 254, 255]
    node_names = np.array(['150.1', '150.2', '150.3', '150r.1', '150r.2', '150r.3', '149.1', '149.2', '149.3', '1.1', '1.2', '1.3', '2.2', '3.3', '7.1', '7.2', '7.3', '4.3', '5.3', '6.3', '8.1', '8.2', '8.3', '12.2', '9.1', '13.1', '13.2', '13.3', '9r.1', '14.1', '34.3', '18.1', '18.2', '18.3', '11.1', '10.1', '15.3', '16.3', '17.3', '19.1', '21.1', '21.2', '21.3', '20.1', '22.2', '23.1', '23.2', '23.3', '24.3', '25.1', '25.2', '25.3', '25r.1', '25r.3', '26.1', '26.3', '28.1', '28.2', '28.3', '27.1', '27.3', '31.3', '33.1', '29.1', '29.2', '29.3', '30.1', '30.2', '30.3', '250.1', '250.2', '250.3', '32.3', '35.1', '35.2', '35.3', '36.1', '36.2', '40.1', '40.2', '40.3', '37.1', '38.2', '39.2', '41.3', '42.1', '42.2', '42.3', '43.2', '44.1', '44.2', '44.3', '45.1', '47.1', '47.2', '47.3', '46.1', '48.1', '48.2', '48.3', '49.1', '49.2', '49.3', '50.1', '50.2', '50.3', '51.1', '51.2', '51.3', '151.1', '151.2', '151.3', '52.1', '52.2', '52.3', '53.1', '53.2', '53.3', '54.1', '54.2', '54.3', '55.1', '55.2', '55.3', '57.1', '57.2', '57.3', '56.1', '56.2', '56.3', '58.2', '60.1', '60.2', '60.3', '59.2', '61.1', '61.2', '61.3', '62.1', '62.2', '62.3', '63.1', '63.2', '63.3', '64.1', '64.2', '64.3', '65.1', '65.2', '65.3', '66.1', '66.2', '66.3', '67.1', '67.2', '67.3', '68.1', '72.1', '72.2', '72.3', '97.1', '97.2', '97.3', '69.1', '70.1', '71.1', '73.3', '76.1', '76.2', '76.3', '74.3', '75.3', '77.1', '77.2', '77.3', '86.1', '86.2', '86.3', '78.1', '78.2', '78.3', '79.1', '79.2', '79.3', '80.1', '80.2', '80.3', '81.1', '81.2', '81.3', '82.1', '82.2', '82.3', '84.3', '83.1', '83.2', '83.3', '85.3', '87.1', '87.2', '87.3', '88.1', '89.1', '89.2', '89.3', '90.2', '91.1', '91.2', '91.3', '92.3', '93.1', '93.2', '93.3', '94.1', '95.1', '95.2', '95.3', '96.2', '98.1', '98.2', '98.3', '99.1', '99.2', '99.3', '100.1', '100.2', '100.3', '450.1', '450.2', '450.3', '197.1', '197.2', '197.3', '101.1', '101.2', '101.3', '102.3', '105.1', '105.2', '105.3', '103.3', '104.3', '106.2', '108.1', '108.2', '108.3', '107.2', '109.1', '300.1', '300.2', '300.3', '110.1', '111.1', '112.1', '113.1', '114.1', '135.1', '135.2', '135.3', '152.1', '152.2', '152.3', '160r.1', '160r.2', '160r.3', '160.1', '160.2', '160.3', '61s.1', '61s.2', '61s.3', '300_open.1', '300_open.2', '300_open.3', '94_open.1', '610.1', '610.2', '610.3'])
    # the title for each load, for example, columns 0 and 1 in data_in are P and Q representing load s1
    load_names = ['s1a', 's2b', 's4c', 's5c', 's6c', 's7a', 's9a', 's10a', 's11a', 's12b', 
                's16c', 's17c', 's19a', 's20a', 's22b', 's24c', 's28a', 's29a', 's30c', 
                's31c', 's32c', 's33a', 's34c', 's35a', 's37a', 's38b', 's39b', 's41c', 
                's42a', 's43b', 's45a', 's46a', 's47', 's48', 's49a', 's49b', 's49c', 
                's50c', 's51a', 's52a', 's53a', 's55a', 's56b', 's58b', 's59b', 's60a', 
                's62c', 's63a', 's64b', 's65a', 's65b', 's65c', 's66c', 's68a', 's69a', 
                's70a', 's71a', 's73c', 's74c', 's75c', 's76a', 's76b', 's76c', 's77b', 
                's79a', 's80b', 's82a', 's83c', 's84c', 's85c', 's86b', 's87b', 's88a', 
                's90b', 's92c', 's94a', 's95b', 's96b', 's98a', 's99b', 's100c', 's102c', 
                's103c', 's104c', 's106b', 's107b', 's109a', 's111a', 's112a', 's113a', 's114a']
    with open(r'New Data\infomap_ieee123.txt', 'r') as file:
        document_content = file.read()
    clusters = parse_cluster_info(document_content, load_indices)
    '''
        for cluster in clusters:
        print(f'cluster {cluster}: \n{clusters[cluster][0]}\n{clusters[cluster][1]}\n{clusters[cluster][2]}')
        
    '''


    input_df = pd.read_csv("New Data\data_in.csv")
    output_df = pd.read_csv("New Data\data_out.csv")



    cluster_in_5 = filter_input_columns(input_df, clusters['5'][1])
    cluster_out_5 = filter_output_columns(output_df, clusters['5'][0], add_offset=278)
    cluster_in_1 = filter_input_columns(input_df, clusters['1'][1])
    cluster_in_1.to_csv(r'cluster_in_1.csv', index = None, header=None)
    cluster_out_1 = filter_output_columns(output_df, clusters['1'][0], add_offset=278)
    cluster_out_1.to_csv(r'cluster_out_1.csv', index = None, header=None)


    i=0
    print("cluster 5:\n")
    for index in clusters['5'][0]:
        print(f'({i})index={index}, node name={node_names[index]}, phase={output_df.iat[0,index+278]}')
        i+=1
    i=0
    print("cluster 1:\n")
    for index in clusters['1'][0]:
        print(f'({i})index={index}, node name={node_names[index]}, phase={output_df.iat[0,index+278]}')
        i+=1


    admittance_df = pd.read_csv('New Data\ieee123_y_matrix.csv', index_col=0, header=0)
    admittance_df = admittance_df.applymap(parse_complex)
    print(admittance_df[:20])
    print(admittance_df.dtypes)  # Should show complex dtype
    print(type(admittance_df.iloc[0, 0]))  # Should be complex

    # Get the specific row for node 197.1
    admittance_row = admittance_df.loc['197.2']
    print(admittance_row)
    # Compute power for a specific node
        # Compute power for a specific Vi index
    power_df = compute_node_power(
        voltages_df=output_df,
         admittance_row=admittance_row,         target_node_index=230,  # Specify the index for Vi
        node_names=node_names.tolist(),
        neighbors=['101.1', '101.2', '101.3']
    )
    # Print out details to verify alignment
    print("cluster_in_1 shape:", cluster_in_1.shape)
    print("power_df shape:", power_df.shape)
    print(power_df)
    # Concatenate the power columns to cluster_in_1
    cluster_in_1_with_power = pd.concat([cluster_in_1, power_df], axis=1)

    # Save to CSV
    cluster_in_1_with_power.to_csv(r'cluster_in_1_withp.csv', index=None, header=None)


    #train_and_evaluate_nn(cluster_datasets['5'][0], cluster_datasets['5'][1])
    #cluster_out = filter_output_columns(output_df, clusters['5'][0], add_offset=278)
    #print(cluster_out.head())
    #print(cluster_out.shape)
    #cluster_in = filter_input_columns(input_df, clusters['5'][1])
    #print(cluster_in.head())
    #print(cluster_in.shape)
    #cluster_in.to_csv(r'cluster_in_5.csv', index = None, header=None)
    #cluster_out.to_csv(r'cluster_out_5.csv', index = None, header=None)
    #train_and_evaluate_nn(cluster_in, cluster_out, hidden_size=4, lr=0.001)
    #train_simple_nn(input_df=cluster_in, output_df=cluster_out)
    #model, normalizers = train_simple_nn(
    #input_df=cluster_in, 
    #output_df=cluster_out, 
    #amplitude_cols=(0, 26),  # For columns 0-25
    #phase_cols=(26, 52)      # For columns 26-51
    #)
    #simple_output_visualization(output_df)

    #find_value_ranges(cluster_out)
    #find_value_ranges(cluster_in)

main()