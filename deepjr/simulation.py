import numpy as np
import matplotlib.pyplot as plt
import mne
import xarray as xr
from tqdm.notebook import tqdm
from copy import copy
import pandas as pd
import seaborn as sns
from pathlib import Path
from frozendict import frozendict

from mne.simulation import SourceSimulator


# Default Parameters from Jansen RIT model
jr_typical_param = frozendict({
    "A_e": 3.25 * 1e-3,
    "A_i": 22 * 1e-3,
    "b_e": 100,
    "b_i": 50,
    "C": 135,
    "a_1": 1.0,
    "a_2": 0.8,
    "a_3": 0.25,
    "a_4": 0.25,
    "v_max": 50 * 1e-3,
    "v_0": 6 * 1e-3
})

jr_param_ranges = frozendict({
    'A_e': (2.6 * 1e-3, 9.75 * 1e-3),
    'A_i': (17.6 * 1e-3, 110.0 * 1e-3),
    'b_e': (5, 150),
    'b_i': (25, 75),
    'C': (65, 1350),
    'a_1': (0.5, 1.5),
    'a_2': (0.4, 1.2),
    'a_3': (0.125, 0.375),
    'a_4': (0.125, 0.375)
})


class JRSimulationError(RuntimeError):
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)


def get_fsaverage_fwd(info):
    fwd_fname = "fsaverage-fwd.fif.gz"
    if not Path(fwd_fname).exists():
        src = mne.setup_source_space('fsaverage', spacing='oct6',
                                     surface='white')
        model = mne.make_bem_model(subject="fsaverage")
        bem = mne.make_bem_solution(model)
        fwd = mne.make_forward_solution(info, "fsaverage", src, bem)
        fwd.save(fwd_fname)

    return mne.read_forward_solution(fwd_fname)


def generate_events(inter_trial_freq, dt, t_base, t_tot):
    """
    Generate the event array as expected by MNE.

    - inter_trial_freq: Frequency between different repetition of
                        the stimulus (Hz)
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
    - pulse_width_fraction: Fraction of the cycle that the pulse is
                            high (default is 0.05)
    - return_time: Whether to return the time variable.

    Returns:
    - I: Current stimulus array
    - Ip, Ii: Modified versions of the stimulus for different inputs
    - t: (optional) time

    The stimulus arrays (I, Ip, Ii) are designed to drive different aspects of
    the simulated system, affecting different state variables.
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


def run_jr_simulation(dt, Ii, Ip, p, params):
    """
    Run the neural dynamics simulation.

    Parameters:
    - dt: Time step (s)
    - Ii, Ip: Modified stimulus inputs
    - p: Noise term array
    - Ip, Ii: Modified versions of the stimulus for different inputs

    Returns:
    - y: Array of state variables over time

    Membrane potentials (y[0], y[1], y[2]): Typically measured in
                                            millivolts (mV).
    Derivatives (y[3], y[4], y[5]): Measured in millivolts per second (mV/s),
                                    representing the rate of change of the
                                    potentials.
    """

    param_ = dict(jr_typical_param)
    param_.update(params)

    nb_samples = len(p)
    y = np.zeros((6, nb_samples))
    # JR Parameters
    A = param_['A_e']
    B = param_['A_i']
    a = param_['b_e']
    b = param_['b_i']
    C1 = param_['a_1']
    C2 = param_['a_2']
    C3 = param_['a_3']
    C4 = param_['a_4']
    v0 = param_['v_0']
    vm = param_['v_max']
    r = 0.56  # Need to be defined
    ka = 1  # Need to be defined
    kA = 1  # Need to be defined

    for ii in range(1, nb_samples):
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


def calculate_mean_std(lo, hi):
    """Calculate mean and standard deviation from the given range."""
    mean = (hi + lo) / 2
    std = (hi - lo) / 4
    return mean, std


def sample_jansen_rit_parameters(parameter_ranges, num_samples,
                                 method='linear'):
    """
    Samples ground truth values for Jansen-Rit model parameters.

    Parameters:
    - parameter_ranges: dict, contains parameter names as keys and a tuple of
                        (low, high) range as values
    - num_samples: int, number of samples to generate
    - method: str, sampling method ('linear' for linear spacing, 'normal'
              for normal distribution)

    Returns:
    - sampled_params: dict, contains arrays of sampled values
                      for each parameter
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
            raise ValueError("Unknown sampling method. "
                             "Use 'linear' or 'normal'.")

    return pd.DataFrame(sampled_params)


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
    return evoked_xarray.transpose("sim_no", "time", "ch_names")


def linear_spread_noise_fact(start, stop, num):
    return np.linspace(start, stop, num).round(2).tolist()


def apply_C_factor(parameters, inplace=False):

    if isinstance(parameters, frozendict):
        if inplace:
            raise RuntimeError("apply_C_factor cannot be performed "
                               "in-place with a frozendict")
        parameters = dict(parameters)

    ret_dict = parameters if inplace else copy(parameters)
    for i in range(1, 5):
        ret_dict[f"a_{i}"] *= parameters["C"]
    return ret_dict


class EventRelatedExp:

    def __init__(self, info, isi=1, nb_trial=60,
                 stim_dur=0.05, transient_dur=0.8,
                 epochs_tmin=-0.2):
        """
            isi: inter-stimulus interval, in seconds
            nb_trial: Number of trials
            stim_dur: Stimulus duration, in seconds
            transient_dur: Duration of transient
        """
        self.epochs_tmin = epochs_tmin
        self.isi = isi
        self.stim_dur = stim_dur
        self.transient_dur = transient_dur
        self.sim_duration = transient_dur - epochs_tmin + nb_trial*isi
        self.events = generate_events(inter_trial_freq=1/isi,
                                      dt=1/info["sfreq"],
                                      t_base=transient_dur - epochs_tmin,
                                      t_tot=self.sim_duration)

        # Calculate the stimulus signal
        stim_data = generate_stimulus(1/info["sfreq"],
                                      self.sim_duration,
                                      self.events,
                                      return_time=True)
        self.I, self.Ip, self.Ii, self.time = stim_data

    def plot_stimulus(self):
        plt.plot(self.time, self.I)

    def get_epochs_from_raw(self, raw, tmin=None, tmax=1,
                            baseline=(None, 0), verbose=None):
        if tmin is None:
            tmin = self.epochs_tmin

        return mne.Epochs(raw, self.events, tmin=tmin,
                          tmax=tmax, baseline=baseline, verbose=verbose)

    @property
    def nb_samples(self):
        return len(self.Ip)


class JRSimulator:

    def __init__(self, head_model_kwargs=None):
        if head_model_kwargs is None:
            head_model_kwargs = {}
        self.set_head_model(**head_model_kwargs)

    def set_head_model(self, sfreq=1000, montage='biosemi64',
                       subject="fsaverage", subjects_dir=None):
        if isinstance(montage, str):
            self.montage = mne.channels.make_standard_montage(montage)
        else:
            self.montage = montage

        self.info = mne.create_info(self.montage.ch_names, sfreq,
                                    ch_types="eeg")
        self.info.set_montage(self.montage)
        self.noise_cov = mne.make_ad_hoc_cov(self.info)

        self.subject = subject
        self.subjects_dir = subjects_dir
        if subject == "fsaverage":
            self.fwd = get_fsaverage_fwd(self.info)
        else:
            raise NotImplementedError

    @property
    def dt(self):
        return 1.0/self.info['sfreq']

    def run_simulation(self, experiment, parameters, jr_noise_sd=0.0,
                       jr_noise=None, exclude_transient=True, apply_C=True):

        self.experiment = experiment
        if jr_noise is None:
            jr_noise = jr_noise_sd * np.random.randn(experiment.nb_samples)
        #if exclude_transient:
        #    points_to_skip = int(experiment.transient_dur/self.dt)
        #else:
        #    points_to_skip = 0

        if apply_C:
            parameters = apply_C_factor(parameters)
        y = run_jr_simulation(self.dt, experiment.Ii, experiment.Ip,
                              jr_noise, parameters)
        self.y = y  # [:, points_to_skip:]
        self.time = experiment.time  # [points_to_skip:]
        self.output = (y[1] - y[2])  # [points_to_skip:]

    def plot_jr_results(self):
        """
        Plot the state variables and outputs.
        """

        fig, axes = plt.subplots(7, 1, figsize=(10, 10), sharex=True)

        # Plot each state variable in a separate subplot
        for j, ax, ax_res in zip(range(1, 7), axes, self.y):
            ax.plot(self.time, ax_res)
            ax.set_ylabel(f'State {j} [V]')

        # Output plot on the last subplot
        ax = axes[6]
        ax.plot(self.time, self.output)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Output (y[1]-y[2]) [V]')

        fig.tight_layout()

        return fig, axes

    def simulate_for_parameter(self, experiment, parameter_ranges=None,
                               nb_sims=1,
                               ground_truth=None, method=None,
                               noise_fact=None,
                               base_path=None, use_tqdm=False,
                               seed=0, save=True,
                               jr_noise=None, jr_noise_sd=0.0,
                               verbose=None):
        """
        Simulates and saves results for a specific parameter range.

        :param parameter_ranges: Key for the parameter to simulate and Range
                                 tuple (low, high) for the parameter.
        :param N: Number of samples to generate.
        :param ground_truth: Dictionary of ground truth values for JR
                             parameters.
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

        def __prepare_params(ground_truth, sampled_values_row):
            # Start with the ground truth values
            jr_params = ground_truth.copy()
            for param in parameter_ranges:
                if param in sampled_values_row:
                    jr_params[param] = sampled_values_row[param]
            return jr_params

        np.random.seed(seed)

        if parameter_ranges is None:
            parameter_ranges = dict(jr_param_ranges)

        if ground_truth is None:
            ground_truth = dict(jr_typical_param)

        nb_samples = experiment.nb_samples
        if jr_noise is None:
            jr_noise = jr_noise_sd * np.random.randn(nb_samples)

        # Data Preparation
        sampled_values = sample_jansen_rit_parameters(parameter_ranges,
                                                      num_samples=nb_sims,
                                                      method=method)
        # Run the simulations
        iter = sampled_values.iterrows()
        if use_tqdm:
            iter = tqdm(list(iter), leave=False)

        evoked_results = []
        snr_db_output = []
        for _, row in iter:
            jr_params = __prepare_params(ground_truth, row)
            try:
                self.run_simulation(experiment, jr_params, jr_noise_sd,
                                    apply_C=True)
                self.generate_raw(seed=np.random.randint(2**32 - 1),
                                  noise_fact=noise_fact, verbose=verbose)
                self.generate_evoked(experiment)

                evoked_results.append(self.evoked)
                snr_db_output.append(self.raw_snr)
            except JRSimulationError:
                evoked_results.append(None)
                snr_db_output.append(np.nan)

        self.sim_results = SimResults(nb_sims, noise_fact, base_path)
        self.sim_results.make_dataset(evoked_results, snr_db_output,
                                      sampled_values)

        if save:
            self.sim_results.save()

    def compute_raw_from_output(self, output,
                                target_region="caudalmiddlefrontal-lh",
                                location="center", extent=10.0, gain=1e-6,
                                seed=0, noise_fact=None,
                                verbose=None):
        """
        location: Use the center of the region as a seed.
        extent: Extent in mm of the region.
        """

        # select a region to activate, we use the caudal middle frontal to grow
        # a region of interest.
        label_kwargs = dict(regexp=target_region,
                            subjects_dir=self.subjects_dir,
                            verbose=verbose)
        selected_label = mne.read_labels_from_annot(self.subject,
                                                    **label_kwargs)[0]

        label = mne.label.select_sources(self.subject, selected_label,
                                         location=location, extent=extent)

        # Create simulated source activity. Here we use a SourceSimulator whose
        # add_data method is key. It specified where (label), what
        # (source_time_series), and when (events) an event type will occur.

        source_simulator = SourceSimulator(self.fwd["src"],
                                           tstep=self.dt)
        source_simulator.add_data(label,
                                  gain*np.array(output),
                                  [[0, 0, 0]])

        # Project the source time series to sensor space and add some noise.
        # The source simulator can be given directly to the simulate_raw
        # function.
        raw = mne.simulation.simulate_raw(self.info, source_simulator,
                                          forward=self.fwd,
                                          verbose=verbose)

        raw.set_eeg_reference(projection=True, verbose=verbose)

        raw_clean = raw.copy()

        if noise_fact:
            noise_cov = self.noise_cov.copy()
            noise_cov['data'] = noise_cov.data * noise_fact
            # Add noise using the provided noise covariance
            # matrix X by noise factor
            mne.simulation.add_noise(raw, cov=noise_cov,
                                     random_state=seed)
                
            signal = np.mean(raw_clean.get_data()**2)
            noise = np.mean((raw.get_data() - raw_clean.get_data())**2)
            raw_snr = 10*np.log10(signal/noise)
        else:
            raw_snr = np.nan

        raw.pick('eeg')

        annotations = mne.annotations_from_events(self.experiment.events,
                                                  self.info["sfreq"],
                                                  {0: "stim"},
                                                  verbose=False)
        raw.set_annotations(annotations, verbose=False)

        return raw, raw_snr

    def generate_raw(self, **kwargs):
        """
        location: Use the center of the region as a seed.
        extent: Extent in mm of the region.
        """

        self.raw, self.raw_snr = self.compute_raw_from_output(self.output,
                                                              **kwargs)

    def generate_evoked(self, experiment, **epoching_kwargs):
        self.epochs = experiment.get_epochs_from_raw(self.raw,
                                                     **epoching_kwargs)
        self.evoked = self.epochs.average()


def get_non_outliers_sim_no(x, k=3):
    x = np.abs(x).mean(dim=["time", "ch_names"])

    q1, q3 = x.quantile([0.25, 0.75])

    th_min = q1 - k*(q3-q1)
    th_max = q3 + k*(q3-q1)

    return x[(x > th_min) & (x < th_max)].sim_no


class SimResults:

    def __init__(self, nb_sims, noise_fact=0, path="./"):
        self.nb_sims = nb_sims
        self.noise_fact = noise_fact
        if noise_fact:
            noise_fact_str = (str(noise_fact).replace('.', '_')
                              if '.' in str(noise_fact) else noise_fact)
            self.fname = f'xarr_noise_all_{nb_sims}_{noise_fact_str}.nc'
        else:
            self.fname = f'xarr_all_{nb_sims}.nc'
        self.path = path

    def load(self):
        self.dataset = xr.open_dataset(self.path / self.fname)

    def make_dataset(self, evoked_results, snr_db_output, jr_normal_params):
        evoked_xr = get_evoked_xarray(evoked_results)
        snr_xr = xr.DataArray(pd.Series(snr_db_output), dims=["sim_no"])
        param_xr = xr.DataArray(pd.DataFrame(jr_normal_params),
                                dims=["sim_no", "param"])

        self.dataset = xr.Dataset({"evoked": evoked_xr, "snr": snr_xr,
                                   "parameters": param_xr})

    def save(self, overwrite=True):
        # To avoid potential permission errors.
        if overwrite:
            self.full_path.unlink(missing_ok=True)
            self.dataset.to_netcdf(self.full_path)
        elif not self.full_path.exists():
            self.dataset.to_netcdf(self.full_path)
        else:
            raise RuntimeError("File already exists. "
                               "Pass overwrite==True to overwrite.")

    def clean(self):
        self.dataset = self.dataset.dropna("sim_no", subset=["evoked"])

        # Drop outliers
        non_outliers_no = get_non_outliers_sim_no(self.dataset['evoked'])
        self.dataset = self.dataset.sel(sim_no=non_outliers_no)

    def plot_evoked_heatmap(self):
        X = self.dataset["evoked"].squeeze()
        df = pd.DataFrame(X.max("ch_names").values,
                          columns=X.time)
        sns.heatmap(df)

    @property
    def full_path(self):
        return self.path / self.fname

    @property
    def snr(self):
        return pd.Series(self.dataset["snr"].values,
                         index=self.dataset.sim_no)
