###
## cluster_maker
## A package to simulate clusters of data points.
## J. Foadi - University of Bath - 2024
##
## Module dataframe_builder
###

## Libraries needed
import pandas as pd
import numpy as np

## Function to define the wanted data structure
def define_dataframe_structure(column_specs):
    """"
    First an empty dictionary is created which can store column data
    maximum length is initialised to 0 to keep track of the length of points across the columns

    for each spec it updates the max length to be maximum length found

    returns column name for each spec and extends numerical columns with NaN to match max_length

    the extended list is added to data dictionary with column name as the key

    finally the function converts data dictionary to a pandas dataframe which is returned


    """
    # Prepare data dictionary
    data = {}
    max_length = 0

    # Find the maximum length of representative points
    for spec in column_specs:
        max_length = max(max_length, len(spec.get('reps', [])))

    for spec in column_specs:
        name = spec['name']
        reps = spec.get('reps', [])
        # Extend numerical columns with NaN to match max_length
        extended_points = reps + [np.nan] * (max_length - len(reps))
        data[name] = extended_points

    return pd.DataFrame(data)

# cluster_maker/dataframe_builder.py

def export_to_csv(dataframe, file_path):
    dataframe.to_csv(file_path)

## Function to simulate data
def simulate_data(seed_df, n_points=100, col_specs=None, random_state=None):
    """

    seed_df is a dataframe which is seed data used to generate data points

    n_points is the number of data points to be generated

    col_specs is dictionary of specifications for each column

    random_state is optional to sed random seed

    1) after random seed is set for reproducibility, an empty list is created to store simulated data points
    2) We iterate over each row using iterrows()
    3) For each row we generate n points
    4) apply column specific specifications, depending on distributin type the function simulates values
    5) we then apped to simulated data list


    """
    if random_state is not None:
        np.random.seed(random_state)
    
    simulated_data = []

    for _, representative in seed_df.iterrows():
        for _ in range(n_points):
            simulated_point = {}
            for col in seed_df.columns:
                # Numerical columns: apply column-specific specifications
                if col_specs and col in col_specs:
                    dist = col_specs[col].get('distribution', 'normal')
                    variance = col_specs[col].get('variance', 1.0)

                    if dist == 'normal':
                        simulated_point[col] = representative[col] + np.random.normal(0, np.sqrt(variance))
                    elif dist == 'uniform':
                        simulated_point[col] = representative[col] + np.random.uniform(-variance, variance)
                    else:
                        raise ValueError(f"Unsupported distribution: {dist}")
                else:
                    raise ValueError(f"Column {col} has no specifications in col_specs.")
            simulated_data.append(simulated_point)
    
    return pd.DataFrame(simulated_data)

def non_globular_cluster(seed_df, n_points=100, col_specs=None, random_state=None, shape='spiral', noise=0.1):
    """
    Simulate non-globular clusters based on seed data and column specifications.

    Parameters:
    seed_df (pd.DataFrame): Seed data used to generate data points.
    n_points (int): Number of data points to be generated.
    col_specs (dict): Dictionary of specifications for each column.
    random_state (int): Optional random seed for reproducibility.
    shape (str): Shape of the non-globular cluster ('spiral', 'crescent', etc.).
    noise (float): Amount of noise to add to the data points.

    Returns:
    pd.DataFrame: A DataFrame containing the simulated non-globular cluster data points.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    simulated_data = []

    for _, representative in seed_df.iterrows():
        for _ in range(n_points):
            simulated_point = {}
            for col in seed_df.columns:
                if col_specs and col in col_specs:
                    base_value = representative[col]
                    if shape == 'spiral':
                        angle = np.random.uniform(0, 2 * np.pi)
                        radius = np.random.uniform(0, 1)
                        x = radius * np.cos(angle)
                        y = radius * np.sin(angle)
                        simulated_point[col] = base_value + x + np.random.normal(0, noise)
                    elif shape == 'crescent':
                        angle = np.random.uniform(0, np.pi)
                        radius = np.random.uniform(0.5, 1)
                        x = radius * np.cos(angle)
                        y = radius * np.sin(angle)
                        simulated_point[col] = base_value + x + np.random.normal(0, noise)
                    else:
                        raise ValueError(f"Unsupported shape: {shape}")
                else:
                    raise ValueError(f"Column {col} has no specifications in col_specs.")
            simulated_data.append(simulated_point)

    return pd.DataFrame(simulated_data)