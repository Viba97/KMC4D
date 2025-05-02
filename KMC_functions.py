#TEST NEW WITH DIFFUSION COEFFICIENT 

import numpy as np

##############################################################################################
####### CONSTANT and CONVERSION CONSTAN ######################################################
##############################################################################################
KB = 1.3806488e-23  # Boltzmann constant
H = 6.62606957e-34  # Planck constant
C = 2.99792458e10  # Speed of light in cm/s
R = 8.3144621  # Ideal gas constant
AV = 6.0221415e23  # Avogadro number
EhtokJmol = 2625.5002
AMUtoKG = 1.66053886e-27
CMtokJmol = 1.1962659192089765e-2  # C * H * AV change due to numerical errors
CMtoK = 1.4387862961655296
kjtokcal = 4.184
kJmoltoK = 120.2731159571
MHZtoK = 6.62606957 / 1.3806488 * 1e-5  # e-34 * 1e6 / e-23
AUconv = 1.66053886 * 1.3806488 / 6.62606957 / 6.62606957 * 1e18  # AMUtoKG * KB / H / H
UMAAAtoKGM = 1.660539e-47
angstrom_to_cm2 = 1e-16  # Convert Å^2 to cm^2
##############################################################################################
##############################################################################################

def q_rot(t, Rot_info, sim, rot_unit):
    if rot_unit:
        return np.sqrt(np.pi) * np.sqrt(np.prod(np.array(Rot_info) * CMtoK * t)) / sim
    else:
        return (np.sqrt(np.pi) / sim * np.power(8 * np.pi**2 * KB * t, 1.5) * 
                np.sqrt(np.prod(np.array(Rot_info) * UMAAAtoKGM)) / H**3)

def lamda_trasl(t, m):
    return H / np.sqrt(2 * np.pi * m * AMUtoKG * KB * t)

def pre_factor_tait(t, m, a, sim, Rot_info, rot_unit):
    lmd = lamda_trasl(t, m)
    qrot = q_rot(t, Rot_info, sim, rot_unit)
    return (KB * t / H) * (a / lmd**2) * qrot

def compute_rate(prefactor, barrier, temperature):
    return prefactor * np.exp(-barrier / (R * temperature))

def log_visited_sites(cycle_count, visit_counts, current_time):
    visited_sites = [site for site, count in visit_counts.items() if count > 0]
    print(f"\n--- Cycle {cycle_count} ---")
    print(f"Number of visited sites: {len(visited_sites)}")
    print(f"Visited sites (BE sites): {visited_sites}")
    print(f"Current simulation time: {current_time:.12f} s")
    print("--------------------------\n")

def calculate_3d_distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2 + (coord1[2] - coord2[2])**2)

def simulate_site_transition(start_site, final_time, final_cycle_count, index_to_site_mapping, 
                             diff_des_barriers, site_names, cumulative_visit_counts, unique_sites, 
                             sulfur_positions, sim_num, site_max_BE, log_interval):
    
    # Initialize simulation time and cycle counter
    current_time = 0
    cycle_count = 0
    
    visit_counts = {int(site): 0 for site in unique_sites}
    positions_site = []
    times_site = []
    desorption_occurred = False
    desorption_site = None  # Track the desorption site
    total_weighted_diffusion = 0
    total_time_step = 0

    print(f"##### STARTING FROM SITE: {start_site}")
                                 
    # Get index of the starting site in the matrix
    starting_site_index = np.where(site_names == float(start_site))[0][0]

    # Main simulation loop: run until time or cycle limits are reached
    while current_time < final_time and cycle_count < final_cycle_count:
        # Get transition rates from current site to all others
        rates_from_start = diff_des_barriers[starting_site_index, :]
        # Get current site name from index
        current_site = index_to_site_mapping[starting_site_index]
        # Sum of all possible transition rates from current site
        sum_of_rates_from_start = np.sum(rates_from_start)

        if sum_of_rates_from_start > 0:
            # Normalize rates to get transition probabilities
            probabilities_from_start = rates_from_start / sum_of_rates_from_start
            random_number = np.random.rand()
            cumulative_probability = 0.0
            selected_event_index = None

            #KMC CORE!!!!
            # Determine which site is selected based on cumulative probability
            for j in range(len(probabilities_from_start)):
                cumulative_probability += probabilities_from_start[j]
                if random_number <= cumulative_probability:
                    selected_event_index = j
                    selected_site = index_to_site_mapping[selected_event_index]
                    break
                    
            # If the selected site is the same as the current, desorption occurred
            if selected_event_index == starting_site_index:
                desorption_occurred = True
                desorption_site = current_site  # Record the site where desorption occurred
                print(f"Desorption occurred at site {current_site} with BE value {round(site_max_BE[str(int(current_site))] / 1000, 2)} kJ/mol at time {current_time:.12f} s.")
                break

            # Calculate displacement between current and next site
            coord_current = sulfur_positions[str(int(current_site))]
            coord_previous = sulfur_positions[str(int(selected_site))]
            displacement = calculate_3d_distance(coord_current, coord_previous)
            squared_displacement = displacement ** 2

            # Sample the time step from exponential distribution
            random_number_for_time = np.random.rand()
            time_step = -np.log(random_number_for_time) / sum_of_rates_from_start

            # Calculate local diffusion coefficient D_i and accumulate weighted value
            #AS DEFINED IN THE PAPER
            D_i = squared_displacement / (2 * 3 * time_step) * angstrom_to_cm2
            total_weighted_diffusion += D_i * time_step
            total_time_step += time_step
            
            current_time += time_step
            cycle_count += 1
            visit_counts[int(unique_sites[selected_event_index])] += 1
            positions_site.append(int(unique_sites[selected_event_index]))
            times_site.append(float(current_time))

            if cycle_count % log_interval == 0:
                log_visited_sites(cycle_count, visit_counts, current_time)

            # Move to the next selected site!!!!
            starting_site_index = selected_event_index
            
        else:
            print("ERROR: No rates available from the starting site.")
            break

    average_diffusion_coefficient = total_weighted_diffusion / total_time_step if total_time_step > 0 else 0
    visited_sites = [site for site, count in visit_counts.items() if count > 0]
    print(f"\n##### SUMMARY FOR SITE {start_site} #####")
    print(f"Number of visited sites: {len(visited_sites)}")
    print(f"Visited site names (BE sites): {visited_sites}")
    print(f"Final simulation time for site {start_site}: {current_time:.12f} s")
    print(f"Number of cycles completed: {cycle_count}")
    print(f"Average diffusion coefficient (cm^2/s): {average_diffusion_coefficient:.12e}")
    print("#########################################\n")

    return {
        "Iteration": sim_num + 1,
        "Start site": start_site,
        "Visited_sites": visited_sites,
        "N_cycles": cycle_count,
        "N_visited_sites": len(visited_sites),
        "Final_time": round(current_time, 6),
        "Desorption": desorption_occurred,
        "Desorption_site": desorption_site,  # Include desorption site in output
        "Average_time_step": round(total_time_step / cycle_count if cycle_count > 0 else 0, 6),
        "Avg_diff_coeff": average_diffusion_coefficient
    }
