import pandas as pd
import numpy as np
from ase.io import read
from KMC_functions import *

#####################################################

R = 8.314  # J/(mol·K)
temperature = 15
final_time = 600.0
final_cycle_count = 10e8
print_interval = 100000

########## CONDITION FOR COMPUTING THE PREFACTOR USING TAIT(2005) ############

A = 1e-19  # Surface per adsorbed molecules
mol = read('/H2S_opt.xyz')
mass = mol.get_masses().sum()
inertia_moments = mol.get_moments_of_inertia()
rot_sim = 2
desorption_prefactor = pre_factor_tait(temperature, mass, A, rot_sim, inertia_moments, False)

########################### DATAFRAME DEFINITION ###################################################################

coordinates_sulphur_BE_site = "/sulfur_positions.txt"
csv_file_path = '/NEW_FINAL.csv'
df_diff_total = pd.read_csv(csv_file_path, index_col=False)

#SELECTING ONLY THE PATH WITH A POSITIVE DIFFUSION BARRIER AND A NEGATIVE EIGENVECTOR ASSOCIATED TO THE TRANSITION STATE
#CONDITION ON THE BE IS NOT NECESSARY BEACUSE THE DATAFRAME WAS CREATED FROM THE PUBLISHED VALUES (BARIOSCO ET AL. 2024, MNRAS)
df_diff_total = df_diff_total[(df_diff_total["react_to_prod_ZPE"] > 0) & (df_diff_total["prod_to_react_ZPE"] > 0) & (df_diff_total["Freq_imm_TS"] < 0)]

#CONNECTED SITES EXCLUDED DUE TO THE LOW CLOSENESS CENTRALITY VALUES
exclude_numbers = ["13_271", "13_432", "271_337", "271_432", "337_432", "26_78"]

df_diff_total = df_diff_total[~df_diff_total['Folder'].isin(exclude_numbers)]


############################### CREATING THE DIFFUSION AND DESORPTION MATRIX ##############################################################

# Extract unique sites
# Initialize a dictionary to store the maximum binding energy (BE) for each site
# WE ADD THIS CONTROL BECAUSE DIFFERENT BE VALUES ARE ASSOCIATED TO A SINGULAR SITE
# THIS IS DUE TO THE DIFFERENT SIZE AND SHAPE OF THE ONIOM MODEL ZONE, BASED ON THE DIFFUSION PATH
site_max_BE = {}
encountered_r_sites = set()
unique_sites = set()
df_diff_total['Folder'].apply(lambda x: unique_sites.update(x.split('_')))
unique_sites = sorted(unique_sites, key=int)
site_to_index = {site: idx for idx, site in enumerate(unique_sites)}

# Define the size of the transition matrix (include extra row and column for site labels)
# USEFUL IN THE DEVELOPMENT PART TO CHECK IF CODE IS WORKING PROPERLY
# NOT REMOVED IN THE END, COULD BE USEFUL
matrix_size = len(unique_sites) + 1

# Initialize a square matrix to hold diffusion and desorption rates
diff_des_barriers = np.zeros((matrix_size, matrix_size))

# Load sulfur positions from the text file
sulfur_positions = {}
with open(coordinates_sulphur_BE_site, 'r') as file:
    for line in file:
        parts = line.split()
        site = parts[0]
        x, y, z = map(float, parts[1:])
        sulfur_positions[site] = (x, y, z)


for index, row in df_diff_total.iterrows():
    r_site, p_site = row['Folder'].split('_')
    
    # Convert site identifiers to matrix indices (offset by 1 due to label row/column)
    r_index = site_to_index[r_site] + 1
    p_index = site_to_index[p_site] + 1
    
    r_to_p_barrier = (row['r_to_p_barrier'] * 0.78) * 1000  #CONVERSION FACTOR FOR THE BARRIER. AS DEFINED IN THE PAPER
    p_to_r_barrier = (row['p_to_r_barrier'] * 0.78) * 1000  #CONVERSION FACTOR FOR THE BARRIER. AS DEFINED IN THE PAPER

    # Calculate the rate prefactor using the transition state frequency and a constant
    prefactor = row['Freq_imm_TS'] * C

    # Calculate the forward and reverse diffusion rates using Arrhenius-like expressions
    r_to_p_rate = -prefactor * np.exp(-r_to_p_barrier / (R * temperature))
    p_to_r_rate = -prefactor * np.exp(-p_to_r_barrier / (R * temperature))
    
    diff_des_barriers[r_index, p_index] = r_to_p_rate
    diff_des_barriers[p_index, r_index] = p_to_r_rate
    current_BE = (row['BE0_post_react'] * 0.76) * 1000      #CONVERSION FACTOR FOR THE BINDING. AS DEFINED IN THE PAPER
    
    # Mark this reactant site as encountered
    encountered_r_sites.add(r_site)

    # Update the maximum binding energy observed for the reactant site
    site_max_BE[r_site] = max(site_max_BE.get(r_site, 0), current_BE)

# Determine sites that were never used as reactants
# DUE TO THE STRUCTURE OF THE DATAFRAME, IT CAN HAPPEN THAT TO SAME BE SITE THERE IS NO BE ASSOAICTED
# THIS IS DUE TO THE FACT THAT THE CODE TRY TO ASSOCIATE THE BE LOOKING FOR THE BE VALUE OF THE REACTANT (LEFT LAEL IN THE NOTATION "X_Y")
# AT THE END OF THE FOR CYCLE THE BE SITE WITHOUT VALUE ARE IDENTIFIED 
missing_sites = set(unique_sites) - encountered_r_sites

# Handle missing sites
for missing_site in missing_sites:
    potential_matches = df_diff_total[df_diff_total['Folder'].str.endswith(f'_{missing_site}')]
    if not potential_matches.empty:
        # Estimate max BE from product state energies
        site_max_BE[missing_site] = (potential_matches['BE0_post_prod'].max() * 0.76) * 1000  #CONVERSION FACTOR FOR THE BINDING. AS DEFINED IN THE PAPER


# Set diagonal elements to the maximum BE(0)_post_react values
for site, max_BE_value in site_max_BE.items():
    site_index = site_to_index[site] + 1
    diff_des_barriers[site_index, site_index] = desorption_prefactor * np.exp(-max_BE_value / (R * temperature))

# Populate first row and column of the matrix with site labels
# USEFUL FOR DEBUGGING
diff_des_barriers[0, 1:] = unique_sites
diff_des_barriers[1:, 0] = unique_sites

unique_sites = diff_des_barriers[0, 1:].astype(int)
matrix_size = len(unique_sites)
site_names = diff_des_barriers[0, 1:]
diff_des_barriers = diff_des_barriers[1:, 1:]
index_to_site_mapping = {i: site_names[i] for i in range(len(site_names))}

cumulative_visit_counts = {int(site): 0 for site in unique_sites}
simulation_results = []

for start_site in unique_sites:
    for sim_num in range(10):
        print(f"--- Simulation {sim_num + 1} for start site {start_site} ---")
        sim_data = simulate_site_transition(start_site, final_time, final_cycle_count, index_to_site_mapping, 
                                            diff_des_barriers, site_names, cumulative_visit_counts, unique_sites, 
                                            sulfur_positions, sim_num, site_max_BE, print_interval)
        simulation_results.append(sim_data)

simulation_df = pd.DataFrame(simulation_results)
simulation_df.to_csv("kmc_simulation_results.csv", index=False)
print("Simulations completed and results saved to CSV.")
