import pandas as pd
import cluster_maker as cm

# Define column specifications
column_specs = [
    {'name': 'A', 'reps': [1, 2, 3]},
    {'name': 'B', 'reps': [4, 5]}
]

# Define the DataFrame structure
df_structure = cm.define_dataframe_structure(column_specs)
print("Defined DataFrame Structure:")
print(df_structure)

# Define seed DataFrame for simulation
seed_df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

# Define column specifications for simulation
col_specs = {
    'A': {'distribution': 'normal', 'variance': 1.0},
    'B': {'distribution': 'uniform', 'variance': 2.0}
}

# Simulate data
simulated_df = cm.simulate_data(seed_df, n_points=5, col_specs=col_specs, random_state=42)
print("Simulated Data:")
print(simulated_df)

# Export simulated data to CSV
output_file_path = 'simulated_data.csv'
cm.export_to_csv(simulated_df, output_file_path)
print(f"Simulated data exported to {output_file_path}")


import pandas as pd
import matplotlib.pyplot as plt
import cluster_maker as cm

def plot_clusters(df, title, filename=None):
    plt.figure(figsize=(8, 6))
    plt.scatter(df['x'], df['y'], alpha=0.6)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

def main():
    # Define column specifications
    column_specs = [
        {'name': 'x', 'reps': [0]},
        {'name': 'y', 'reps': [0]}
    ]

    # Define the DataFrame structure
    df_structure = cm.define_dataframe_structure(column_specs)

    # Define column specifications for simulation
    col_specs = {
        'x': {'distribution': 'normal', 'variance': 1.0},
        'y': {'distribution': 'normal', 'variance': 1.0}
    }

    # Simulate spiral cluster
    spiral_df = cm.non_globular_cluster(df_structure, n_points=500, col_specs=col_specs, shape='spiral', noise=0.1)
    plot_clusters(spiral_df, 'Spiral Cluster', 'spiral_cluster.png')

    # Simulate crescent cluster
    crescent_df = cm.non_globular_cluster(df_structure, n_points=500, col_specs=col_specs, shape='crescent', noise=0.1)
    plot_clusters(crescent_df, 'Crescent Cluster', 'crescent_cluster.png')

    # Export simulated data to CSV
    output_file_path = 'simulated_data.csv'
    cm.export_to_csv(spiral_df, output_file_path)
    print(f"Simulated data exported to {output_file_path}")

if __name__ == "__main__":
    main()