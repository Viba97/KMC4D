import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Constants
R = 8.314  # J/(mol·K)
c = 2.998e10  # Speed of light in cm/s
temperature = 50  # Temperature in Kelvin
final_time = 400.0  # final simulation time in seconds 
print_interval = 100000  # Change this value to print every 'n' cycles


# Path to your CSV file
csv_file_path = '/Users/vittoriobariosco/Documents/work/NEB_TS_diffusion/h2s_diffusion/dataframe/diffusion_revised_optts.csv'  # Path to CSV file

# Read the CSV file
df_diff_total = pd.read_csv(csv_file_path, delimiter='\t', index_col=False)
df_diff_total = df_diff_total[df_diff_total["Freq_imm_TS"] < 0]
exclude_numbers = ["13_271", "13_432", "271_337", "271_432", "337_432", "26_78"]

# Exclude specified rows
df_diff_total = df_diff_total[~df_diff_total['Folder'].isin(exclude_numbers)]

# Extract unique sites
unique_sites = set()
df_diff_total['Folder'].apply(lambda x: unique_sites.update(x.split('_')))
unique_sites = sorted(unique_sites, key=int)

# Create a mapping from site numbers to indices
site_to_index = {site: idx for idx, site in enumerate(unique_sites)}

# Initialize rates array
matrix_size = len(unique_sites) + 1
rates = np.zeros((matrix_size, matrix_size))

# Populate the rates array using the Arrhenius formula
for index, row in df_diff_total.iterrows():
    r_site, p_site = row['Folder'].split('_')
    r_index = site_to_index[r_site] + 1
    p_index = site_to_index[p_site] + 1
    
    r_to_p_barrier = row['r_to_p_barrier'] * 1000
    p_to_r_barrier = row['p_to_r_barrier'] * 1000
    prefactor = row['Freq_imm_TS'] * c
    
    r_to_p_rate = -prefactor * np.exp(-r_to_p_barrier / (R * temperature))
    p_to_r_rate = -prefactor * np.exp(-p_to_r_barrier / (R * temperature))
    
    rates[r_index, p_index] = r_to_p_rate
    rates[p_index, r_index] = p_to_r_rate

# Add index labels for rows and columns
rates[0, 1:] = unique_sites
rates[1:, 0] = unique_sites

unique_sites = rates[0, 1:].astype(int)
matrix_size = len(unique_sites)

def run_simulation(starting_site_index):
    # Initialize visit counts for each site
    visit_counts = {int(site): 0 for site in unique_sites}
    positions_site = []
    times_site = []
    
    starting_site = rates[starting_site_index, 0].astype(int)
    
    current_time = 0.0
    cycle_count = 0
    
    while current_time < final_time:
        sum_of_rates_from_start = np.sum(rates[starting_site_index, 1:])
        probabilities_from_start = rates[starting_site_index, 1:] / sum_of_rates_from_start
        random_number = np.random.rand()
        
        cumulative_probability = 0.0
        selected_event_index = None
        
        for j in range(matrix_size):
            previous_cumulative_probability = cumulative_probability
            cumulative_probability += probabilities_from_start[j]
            if previous_cumulative_probability < random_number <= cumulative_probability:
                selected_event_index = j
                break
        
        visit_counts[int(unique_sites[selected_event_index])] += 1
        positions_site.append(int(unique_sites[selected_event_index]))
        times_site.append(float(current_time))
        
        random_number_for_time = np.random.rand()
        time_step = -np.log(random_number_for_time) / sum_of_rates_from_start
        
        if current_time + time_step > final_time:
            time_step = final_time - current_time
            print(f"ENDED!!! Cycle: {cycle_count}, Current Time: {current_time:.12f} s, Current Site: {starting_site}")
        
        current_time += time_step
        
        if cycle_count % print_interval == 0:
            print(f"Cycle: {cycle_count}, Time step: {time_step:.12f} s, Current Time: {current_time:.12f} s, Current Site: {starting_site}")
        
        starting_site_index = selected_event_index + 1
        starting_site = unique_sites[selected_event_index]
        
        cycle_count += 1
    
    return {
        "visit_counts": {str(k): v for k, v in visit_counts.items()},
        "positions_site": positions_site,
        "times_site": times_site,
        "final_time": float(current_time),
        "cycle_count": cycle_count
    }

# DataFrame to save results
df_results = pd.DataFrame(columns=["Starting_Point", "Most_Visited_Site_1", "Most_Visited_Site_2", "Total_Visited_Sites"])

# Run simulations for each unique site
for start_site in unique_sites:
    print(f"Starting simulation from site: {start_site}")
    starting_site_index = np.where(rates[0, :] == start_site)[0][0]
    result = run_simulation(starting_site_index)
    
    visit_counts = {int(k): v for k, v in result["visit_counts"].items()}
    sorted_visits = sorted(visit_counts.items(), key=lambda item: item[1], reverse=True)
    
    most_visited_site_1 = sorted_visits[0][0] if len(sorted_visits) > 0 else None
    most_visited_site_2 = sorted_visits[1][0] if len(sorted_visits) > 1 else None
    total_visited_sites = len([site for site in visit_counts if visit_counts[site] > 0]) - 1  # Exclude the starting site
    
    # Create a DataFrame for the current result
    df_current_result = pd.DataFrame({
        "Starting_Point": [start_site],
        "Most_Visited_Site_1": [most_visited_site_1],
        "Most_Visited_Site_2": [most_visited_site_2],
        "Total_Visited_Sites": [total_visited_sites]
    })
    
    # Concatenate the current result to the main DataFrame
    df_results = pd.concat([df_results, df_current_result], ignore_index=True)

# Save the results to a CSV file
df_results.to_csv('simulation_results.csv', index=False)

print("Simulations completed and results saved to CSV.")
