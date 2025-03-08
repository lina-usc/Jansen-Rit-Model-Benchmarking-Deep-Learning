from tqdm import tqdm
import torch
# Updated simulator function that can handle a batch of parameters
def simulator(parameters_batch, parameter_names, jr_sim, er_exp, noise_fact):
    results = []
    for parameters in tqdm(parameters_batch, desc="Simulating", unit="batch"):  # Add tqdm to track progress
        # Convert the tensor to a dictionary
        parameter_dict = {key: float(value) for key, value in zip(parameter_names, parameters)}


        # Run the simulation
        jr_sim.run_simulation(er_exp, parameter_dict, jr_noise_sd=0.0)

        # Ensure `raw` data is generated
        jr_sim.generate_raw(seed=0, noise_fact=noise_fact)  # This should create `jr_sim.raw`

        # Generate evoked data
        jr_sim.generate_evoked(er_exp)

        # Extract the data to be used by SBI
        evoked = jr_sim.evoked.data

        results.append(torch.tensor(evoked, dtype=torch.float32))

    return torch.stack(results)
