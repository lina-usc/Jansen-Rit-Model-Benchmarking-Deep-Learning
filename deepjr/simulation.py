import numpy as np
import matplotlib.pyplot as plt
import mne
import xarray as xr
from tqdm.notebook import tqdm
from copy import copy
import pandas as pd


# Default Parameters from Jansen RIT model
jr_typical_param = {"A_e": 3.25 * 1e-3,
              "A_i": 22 * 1e-3,
              "b_e": 100,
              "b_i": 50,
              "C" : 135,
              "a_1": 1.0,
              "a_2": 0.8,
              "a_3": 0.25,
              "a_4": 0.25,
              "v_max": 50 * 1e-3,
              "v_0": 6 * 1e-3}

jr_param_ranges = {
    'A_e': (2.6 * 1e-3, 9.75 * 1e-3),
    'A_i': (17.6 * 1e-3, 110.0 * 1e-3),
    'b_e': (5, 150),
    'b_i': (25, 75),
    'C': (65, 1350),
    'a_1': (0.5, 1.5),
    'a_2': (0.4, 1.2),
    'a_3': (0.125, 0.375),
    'a_4': (0.125, 0.375)
}


class JRSimulationError(RuntimeError):
    def __init__(self, message):            
        # Call the base class constructor with the parameters it needs
        super().__init__(message)


def generate_events(inter_trial_freq, dt, t_base, t_tot):
    """
    Generate the event array as expected by MNE.

    - inter_trial_freq: Frequency between different repetition of the stimulus (Hz)
    - dt: Time step, i.e., inverse of the sampling frequency. (s)
    - t_base: Baseline time before stimulus starts (s)
    - t_tot: Total simulation time (s)

    Returns:
      - events: MNE event structure
    """

    event_samples = (np.arange(t_base, t_tot, 1/inter_trial_freq)/dt)
    return np.array([event_samples,
                     np.zeros_like(event_samples),
                     np.zeros_like(event_samples)]).T.astype(int)


def generate_stimulus(dt, t_tot, events,
                      pulse_width_fraction=0.05,
                      return_time=False):
    """
    Generate the stimulation array with a pulse function.

    Parameters:
    - dt: Time step (s)
    - t_tot: Total simulation time (s)
    - events: MNE event array.
    - pulse_width_fraction: Fraction of the cycle that the pulse is high (default is 0.05)
    - return_time: Whether to return the time variable.

    Returns:
    - I: Current stimulus array
    - Ip, Ii: Modified versions of the stimulus for different inputs
    - t: (optional) time
    
    The stimulus arrays (I, Ip, Ii) are designed to drive different aspects of the simulated system, affecting different state variables.
    """
    t = np.arange(0, t_tot, dt)
    I = np.zeros_like(t)
    n_samp_stim = int(np.round(1/dt*pulse_width_fraction))
    for ind in events[:, 0]:
        I[ind:ind+n_samp_stim] = 1

    Ip = 60 * I  # Modified input for p
    Ii = 60 * 0.56 * I  # Modified input for i, using r = 0.56

    if return_time:
        return I, Ip, Ii, t
    return I, Ip, Ii


def run_jr_simulation(dt, L, Ii, Ip, p, params):
    """
    Run the neural dynamics simulation.

    Parameters:
    - dt: Time step (s)
    - L: Length of the simulation array
    - Ii, Ip: Modified stimulus inputs
    - p: Noise term array
    - Ip, Ii: Modified versions of the stimulus for different inputs

    Returns:
    - y: Array of state variables over time

    Membrane potentials (y[0], y[1], y[2]): Typically measured in millivolts (mV).
    Derivatives (y[3], y[4], y[5]): Measured in millivolts per second (mV/s), 
                                    representing the rate of change of the potentials.
    """

    p = copy(jr_typical_param)
    p.update(params)

    nb_samples = len(p)
    y = np.zeros((6, nb_samples))
    # JR Parameters
    A = p['A_e']
    B = p['A_i']
    a = p['b_e']
    b = p['b_i']
    C1 = p['a_1']
    C2 = p['a_2'] 
    C3 = p['a_3'] 
    C4 = p['a_4'] 
    v0 = p['v_0']
    vm = p['v_max'] 
    r = 0.56  # Need to be defined
    ka = 1  # Need to be defined
    kA = 1  # Need to be defined

    for ii in range(1, L):
        y[0, ii] = y[0, ii - 1] + dt * y[3, ii - 1]
        y[1, ii] = y[1, ii - 1] + dt * y[4, ii - 1]
        y[2, ii] = y[2, ii - 1] + dt * y[5, ii - 1]
        y[3, ii] = y[3, ii - 1] + dt * (A * a * (Ii[ii - 1] + vm / (1 + np.exp(r * (v0 - (y[1, ii - 1] - y[2, ii - 1]))))) - 2 * a * y[3, ii - 1] - a**2 * y[0, ii - 1])
        y[4, ii] = y[4, ii - 1] + dt * (kA * ka * A * a * (p[ii - 1] + Ip[ii - 1] + C2 * vm / (1 + np.exp(r * (v0 - C1 * y[0, ii - 1])))) - 2 * ka * a * y[4, ii - 1] - ka**2 * a**2 * y[1, ii - 1])
        y[5, ii] = y[5, ii - 1] + dt * (B * b * (Ip[ii - 1] + C4 * vm / (1 + np.exp(r * (v0 - C3 * y[0, ii - 1])))) - 2 * b * y[5, ii - 1] - b**2 * y[2, ii - 1])

    if np.any(np.isnan(y)):
        raise JRSimulationError("JR simultation generation NaN values.")
    if np.any(np.isinf(y)):
        raise JRSimulationError("JR simultation generation infinite values.")

    return y


def plot_jr_results(time_axis, results, outputs, points_to_skip):
    """
    Plot the state variables and outputs.

    Parameters:
    - time_axis: Time axis for plotting (s)
    - results: List containing arrays of state variables
    - outputs: List containing output variable (v2 - v3)
    - points_to_skip: Number of points to skip to remove the transient
    """
    fig, axes = plt.subplots(7, 1, figsize=(10, 10), sharex=True)
    adjusted_time_axis = time_axis[points_to_skip:]  # Adjust time axis to skip transient

    # Iterate through each result set 
    for result, output in zip(results, outputs):
        # Plot each state variable in a separate subplot
        for j, ax, ax_res in zip(range(1, 7), axes, result):
            ax.plot(adjusted_time_axis, ax_res)
            ax.set_ylabel(f'State {j} [mV]')
        
        # Output plot on the last subplot
        ax = axes[6]
        ax.plot(adjusted_time_axis, output)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Output (v2-v3) [mV]')
        
    fig.tight_layout()

    return fig, axes


def generate_evoked_from_jr(info, fwd, outputs, events, tmin=-0.2,
                            tmax=1, noise_factor=None, noise_cov=None,
                            subject="fsaverage", subjects_dir=None,
                            return_raw=False, verbose=None):

    # select a region to activate, we use the caudal middle frontal to grow
    # a region of interest.
    selected_label = mne.read_labels_from_annot(subject, 
                                                regexp="caudalmiddlefrontal-lh",
                                                subjects_dir=subjects_dir,
                                                verbose=verbose)[0]

    location = "center"  # Use the center of the region as a seed.
    extent = 10.0  # Extent in mm of the region.
    label = mne.label.select_sources(
        subject, selected_label, location=location, extent=extent, )

    # Create simulated source activity. Here we use a SourceSimulator whose
    # add_data method is key. It specified where (label), what
    # (source_time_series), and when (events) an event type will occur.

    source_simulator = mne.simulation.SourceSimulator(fwd["src"], tstep=1/info["sfreq"])
    # 1e-9 is for adjusting the output to uV
    source_simulator.add_data(label,  1e-9*np.array(outputs), [[0, 0, 0]])

    # Project the source time series to sensor space and add some noise.
    # The source simulator can be given directly to the simulate_raw function.
    raw = mne.simulation.simulate_raw(info, source_simulator,
                                      forward=fwd, verbose=verbose)

    raw.set_eeg_reference(projection=True, verbose=verbose)

    raw_clean = raw.copy()

    if noise_factor:
        noise_cov['data'] = noise_cov.data * noise_factor
        # Add noise using the provided noise covariance matrix X by noise factor
        mne.simulation.add_noise(raw, cov=noise_cov, random_state=0)
    else:
        # Handle invalid noise values
        print('Function failed generate_evoked_from_jr.. check the noise value')

    signal = np.mean(raw_clean.get_data()**2)
    noise = np.mean((raw.get_data() - raw_clean.get_data())**2)
    snr_db = 10*np.log10(signal/noise)

    raw.pick('eeg')

    epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, baseline=(None, 0))
    evoked = epochs.average()

    if return_raw:
        return evoked, snr_db, raw
    return evoked, snr_db


def calculate_mean_std(lo, hi):
    """Calculate mean and standard deviation from the given range."""
    mean = (hi + lo) / 2
    std = (hi - lo) / 4
    return mean, std


def sample_jansen_rit_parameters(parameter_ranges, num_samples, method=None):
    """
    Samples ground truth values for Jansen-Rit model parameters.

    Parameters:
    - parameter_ranges: dict, contains parameter names as keys and a tuple of (low, high) range as values
    - num_samples: int, number of samples to generate
    - method: str, sampling method ('linear' for linear spacing, 'normal' for normal distribution)

    Returns:
    - sampled_params: dict, contains arrays of sampled values for each parameter
    """
    sampled_params = {}
    for param, (low, high) in parameter_ranges.items():
        if method == 'linear':
            sampled_params[param] = np.linspace(low, high, num_samples)
        elif method == 'normal':
            mean = (high + low) / 2
            std = (high - low) / 4
            sampled_params[param] = np.random.normal(mean, std, num_samples)
        else:
            raise ValueError("Unknown sampling method. Use 'linear' or 'normal'.")

    return sampled_params


# Post-processing and Storage

def get_evoked_xarray(evoked_results):

    inds, evoked_array = zip(*[(i, evoked.get_data()) 
                               for i, evoked in enumerate(evoked_results) 
                               if evoked])

    # times and ch_names are consistent across all evoked objects
    if evoked_results:
        times = evoked_results[0].times
        ch_names = evoked_results[0].ch_names

    coords = {"sim_no": list(inds),
              "ch_names": ch_names,
              "time": times}
    evoked_xarray = xr.DataArray(np.array(evoked_array), coords=coords)

    return evoked_xarray


def linear_spread_noise_fact(start, stop, num):
    return np.linspace(start, stop, num).round(2).tolist()


def apply_C_factor(parameters, inplace=False):
    ret_dict = parameters if inplace else copy(parameters)
    for i in range(1, 5):
        ret_dict[f"a_{i}"] *= parameters["C"]
    return ret_dict


def simulate_for_parameter(parameter_ranges, L, Ii, Ip, p, dt, transient_duration,
                           info, fwd, events,  ground_truth={}, method=None, N=None, 
                           noise_factor=None, noise_cov=None,
                           base_path=None, use_tqdm=False):
    """
    Simulates and saves results for a specific parameter range.

    :param parameter_ranges: Key for the parameter to simulate and Range tuple (low, high) for the parameter.
    :param N: Number of samples to generate.
    :param ground_truth: Dictionary of ground truth values for JR parameters.
    :param dt: Time step for the simulation.
    :param L: Length of the simulation array
    :param Ii, Ip: Modified versions of the stimulus for different inputs
    :param Ip: Pyramidal current.
    :param p: Noise term for JR
    :param C: Scaling constant for JR a_1, a2, a3, a4 parameters.
    :param transient_duration: Time to skip at the start of the simulation.
    :param info: Info object for evoked generation.
    :param fwd: Forward solution for source reconstruction.
    :param events: Events object for evoked generation.
    :param base_path: Base directory path for saving files.
    """

    # Data Preparation
    sampled_values = sample_jansen_rit_parameters(parameter_ranges, num_samples=N, method=method)

    states = []
    outputs = []
    jr_normal_params = []
    evoked_results = []
    # raw_output = []

    snr_db_output = []

    # Run the simulations
    if use_tqdm:
        iter = tqdm(range(N), leave=False)
    else:
        iter = range(N)

    for i in iter:
        jr_params = ground_truth.copy()  # Start with the ground truth values
        for param in parameter_ranges.keys():
            if param in ['a1', 'a2', 'a3', 'a4']:  # Correctly check if param is in the list
                if param in sampled_values:  # Ensure param is in sampled_values before accessing it
                    jr_params[param] = sampled_values['C'][i] * sampled_values[param][i]
            else:
                if param in sampled_values:  # Check for existence in sampled_values before accessing
                    jr_params[param] = sampled_values[param][i]
        jr_normal_params.append(jr_params)     

        try:
            y = run_jr_simulation(dt, L, Ii, Ip, p, jr_params)
            points_to_skip = int(transient_duration / dt)
            six_states = y[:, points_to_skip:]
            states.append(six_states)

            output = (y[1] - y[2])[points_to_skip:]
            outputs.append(output)

            # generate evoked results from the output
            evoked, snr_db = generate_evoked_from_jr(info, fwd, output, events,
                                                     tmin=-0.2, tmax=1,
                                                     noise_factor=noise_factor,
                                                     noise_cov=noise_cov)
            evoked_results.append(evoked)
            snr_db_output.append(snr_db)
            # raw_output.append(raw_g)
        except JRSimulationError:
            evoked_results.append(None)
            snr_db_output.append(np.nan)

    if noise_factor:

        if '.' in str(noise_factor):  
            # Replace '.' with '_' in this element
            noise_factor = str(noise_factor).replace('.', '_')

        #evoked_fname = f'evoked_noise_all_{N}_{noise_factor}-ave.fif.gz'
        xarray_fname = f'xarr_noise_all_{N}_{noise_factor}.nc'
        # raw_file = f'raw_noise_{param_key}_{N}_{noise_factor}.pickle'

    else:
        # Save evoked object with parameter name and N in file name
        #evoked_fname = f'evoked_all_{N}-ave.fif.gz'
        xarray_fname = f'xarr_all_{N}.nc'
        # raw_file = f'raw_{param_key}_{N}.pickle'
        # print(jr_normal_params)

    # save raw
    # with open(base_path + raw_file, 'wb') as file:
    #    pickle.dump(raw_output, file, protocol=pickle.HIGHEST_PROTOCOL)     

    # mne.write_evokeds(base_path / evoked_fname, evoked_results,
    #                  overwrite=True)  # Save evoked results

    evoked_xr = get_evoked_xarray(evoked_results)
    snr_xr = xr.DataArray(pd.Series(snr_db_output), dims=["sim_no"])
    param_xr = xr.DataArray(pd.DataFrame(jr_normal_params), dims=["sim_no", "param"])

    dataset = xr.Dataset({"evoked": evoked_xr, "snr": snr_xr,
                          "parameters": param_xr})

    # To avoid potential permission errors.
    (base_path / xarray_fname).unlink(missing_ok=True)
    dataset.to_netcdf(base_path / xarray_fname)
