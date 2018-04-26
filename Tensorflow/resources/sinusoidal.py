import numpy as np

def sinusoid_data_generation_1D(n_Phi, n_T, omega):
    """
    Creates n_Phi sinusoidal signals with n_T points per signal each with
    angular frequency of omega.
    :param n_Phi: number of partitions of phase interval
    :param n_T: number of partitions of time interval
    :param omega: angular frequency of sinusoids
    :return: phase_range::array, time_range:: array, sinusoids::array
    """
    # Discretization of phase interval [0,2pi]
    phase_range = 2 * np.pi * np.linspace(0, 1, n_Phi)
    # Time range
    time_range = np.linspace(0, 1, n_T)
    # Signals
    sinusoids = np.sin(np.subtract.outer(phase_range, -(omega * time_range)))
    return phase_range, time_range, sinusoids