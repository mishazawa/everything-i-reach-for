import numpy as np

EPS = 1e-6


def create_linear_bin(high, low, bc):
    return np.linspace(low, high, num=bc)


def create_nonlinear_bin(bc):
    u = np.linspace(-1 + EPS, 1 - EPS, num=bc)
    return np.arctanh(u)


def rectify_nonlinear_bin(bc, alpha=0.75):
    u = np.linspace(-1 + EPS, 1 - EPS, num=bc)
    linear_part = u
    warped_part = np.arctanh(u) / np.arctanh(1 - EPS)  # normalize to similar scale
    return alpha * linear_part + (1 - alpha) * warped_part


def scale_bin(b, s):
    return b * s


def discrete_value(input: float, bins: list[float], bin_count) -> float:
    return min(np.digitize(input, bins), bin_count - 1)


def create_bins_for_use(env, bc, scale):
    BIN_COUNT = bc
    SCALE_VEL, SCALE_ANG = scale

    bins_cartpos = create_linear_bin(
        env.observation_space.high[0], env.observation_space.low[0], BIN_COUNT
    )
    bins_poleangle = create_linear_bin(
        env.observation_space.high[2], env.observation_space.low[2], BIN_COUNT
    )
    bins_vel = scale_bin(create_nonlinear_bin(BIN_COUNT), SCALE_VEL)
    bins_angvel = scale_bin(rectify_nonlinear_bin(BIN_COUNT), SCALE_ANG)

    return (bins_cartpos, bins_vel, bins_poleangle, bins_angvel)


def get_moving_avgs(arr, window, convolution_mode):
    """Compute moving average to smooth noisy data."""
    return (
        np.convolve(np.array(arr).flatten(), np.ones(window), mode=convolution_mode)
        / window
    )


def exponential_decay(start_epsilon, final_epsilon, epsilon_decay):
    epsilon = start_epsilon

    def calc():
        nonlocal epsilon
        epsilon = max(final_epsilon, epsilon * epsilon_decay)
        return epsilon

    return calc


def linear_decay(start_epsilon, final_epsilon, epsilon_decay):
    epsilon = start_epsilon

    def calc():
        nonlocal epsilon
        epsilon = max(final_epsilon, epsilon - epsilon_decay)
        return epsilon

    return calc
