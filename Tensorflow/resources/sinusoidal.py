import numpy as np
from itertools import product

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

def sinusoid_from_phase(phases, n_T, omega):
    time_range = np.linspace(0, 1, n_T)
    sinusoids = np.sin(np.subtract.outer(phases, -(omega * time_range)))
    return time_range, sinusoids

def sinusoid_image_phase_combination(phases1, phases2, n_T, omega_values):
    """
    This function produces an array where each row corresponds to a sinusoidal signal with a given phase and
    angular frequency omega. The columns represent the time sampling from the interval [0,1].
    :param phases: Vector with the phases to be used
    :param n_T: Number of elements in the partition of the interval [0,1]
    :param omega: Angular frequency
    :return: np.array with shape (len(phases),n_T)
    """

    # Sampling from phase and space
    space_linspace = np.linspace(0, 1, n_T)
    # Create all possible combinations of phi_1, phi_2
    phase_combinations = np.array(list(product(phases1, phases2)))
    sinusoid_images = np.zeros((n_T, n_T, len(phase_combinations)))

    # Create spatial mesh
    spatial_mesh = np.meshgrid(space_linspace, space_linspace)

    # Generate signals for each combination
    for num_mesh, mesh_dimension in enumerate(spatial_mesh):
        # Omega*dimension
        mesh_expanded_dim = omega_values[num_mesh] * mesh_dimension[:, :, np.newaxis]
        repeated_volume = np.repeat(mesh_expanded_dim, repeats=len(phase_combinations), axis=2)
        # sine(Omega*dimension+phase)
        sinusoid_images += np.sin(np.add(repeated_volume, phase_combinations[:, num_mesh]))
    sinusoid_images = np.swapaxes(sinusoid_images, 2, 0)
    return phase_combinations, sinusoid_images

def sinusoid_image_phase(phases1, phases2, n_T, omega_values):
    """
    This function produces an array where each row corresponds to a sinusoidal signal with a given phase and
    angular frequency omega. The columns represent the time sampling from the interval [0,1].
    :param phases: Vector with the phases to be used
    :param n_T: Number of elements in the partition of the interval [0,1]
    :param omega: Angular frequency
    :return: np.array with shape (len(phases),n_T)
    """

    # Sampling from phase and space
    space_linspace = np.linspace(0, 1, n_T)
    # Create all possible combinations of phi_1, phi_2
    phases1 = np.expand_dims(phases1, 1)
    phases2 = np.expand_dims(phases2, 1)
    phases = np.concatenate((phases1, phases2), axis=1)
    sinusoid_images = np.zeros((n_T, n_T, len(phases)))

    # Create spatial mesh
    spatial_mesh = np.meshgrid(space_linspace, space_linspace)

    # Generate signals for each combination
    for num_mesh, mesh_dimension in enumerate(spatial_mesh):
        # Omega*dimension
        mesh_expanded_dim = omega_values[num_mesh] * mesh_dimension[:, :, np.newaxis]
        repeated_volume = np.repeat(mesh_expanded_dim, repeats=len(phases), axis=2)
        # sine(Omega*dimension+phase)
        sinusoid_images += np.sin(np.add(repeated_volume, phases[:, num_mesh]))
    sinusoid_images = np.swapaxes(sinusoid_images, 2, 0)
    return phases, sinusoid_images