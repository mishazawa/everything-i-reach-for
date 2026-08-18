import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt

from src.utils import get_moving_avgs


def draw_rolling_stats(env, agent, rolling_length=100):
    _, axs = plt.subplots(ncols=3, figsize=(12, 5))

    # Episode rewards (win/loss performance)
    axs[0].set_title("Episode rewards")
    reward_moving_average = get_moving_avgs(env.return_queue, rolling_length, "valid")
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[0].set_ylabel("Average Reward")
    axs[0].set_xlabel("Episode")

    # Episode lengths (how many actions per hand)
    axs[1].set_title("Episode lengths")
    length_moving_average = get_moving_avgs(env.length_queue, rolling_length, "valid")
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[1].set_ylabel("Average Episode Length")
    axs[1].set_xlabel("Episode")

    # Training error (how much we're still learning)
    axs[2].set_title("Training Error")
    training_error_moving_average = get_moving_avgs(agent.training_error, rolling_length, "same")
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    axs[2].set_ylabel("Temporal Difference Error")
    axs[2].set_xlabel("Step")

    plt.tight_layout()
    plt.show()


def draw_bin(bin, color="red"):
    plt.plot([bin[i] for i in range(len(bin))], color=color)
    plt.show()


def draw_actions(agent, bin_count):
    FIXED_CARTPOS_IDX = bin_count // 2  # pick the most common bin from your histogram
    FIXED_VEL_IDX = bin_count // 2  # same idea
    action_grid = np.full((bin_count, bin_count), np.nan)
    value_grid = np.full((bin_count, bin_count), np.nan)

    Q = agent.q_values

    for angle_idx in range(bin_count):
        for angvel_idx in range(bin_count):
            state = (FIXED_CARTPOS_IDX, FIXED_VEL_IDX, angle_idx, angvel_idx)
            if state in Q:  # only plot states actually visited
                q_values = Q[state]
                action_grid[angvel_idx, angle_idx] = Q[state][1] - Q[state][0]
                value_grid[angvel_idx, angle_idx] = np.max(q_values)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axes[0].imshow(action_grid, origin="lower", cmap="coolwarm", aspect="auto")
    axes[0].set_title("Best action (0=left, 1=right)")
    axes[0].set_xlabel("pole angle bin")
    axes[0].set_ylabel("pole angular velocity bin")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(value_grid, origin="lower", cmap="viridis", aspect="auto")
    axes[1].set_title("Max Q-value (state value)")
    axes[1].set_xlabel("pole angle bin")
    axes[1].set_ylabel("pole angular velocity bin")
    fig.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.show()


def draw_visit_count(visit_counts, bin_count):
    BIN_COUNT = bin_count
    FIXED_CARTPOS_IDX = BIN_COUNT // 2  # pick the most common bin from your histogram
    FIXED_VEL_IDX = BIN_COUNT // 2  # same idea
    count_grid = np.full((BIN_COUNT, BIN_COUNT), np.nan)

    for angle_idx in range(BIN_COUNT):
        for angvel_idx in range(BIN_COUNT):
            state = (FIXED_CARTPOS_IDX, FIXED_VEL_IDX, angle_idx, angvel_idx)
            if state in visit_counts:
                count_grid[angvel_idx, angle_idx] = visit_counts[state]

    plt.figure(figsize=(6, 5))
    plt.imshow(count_grid, origin="lower", cmap="magma", aspect="auto")
    plt.title("Visit count")
    plt.xlabel("pole angle bin")
    plt.ylabel("pole angular velocity bin")
    plt.colorbar()
    plt.show()


is_ipython = "inline" in matplotlib.get_backend()

plt.ion()


def plot_durations(episode_durations=None, show_result=False):
    if episode_durations is None:
        episode_durations = []

    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        from IPython import display

        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
