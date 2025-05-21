import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os # For checking file existence
import time # For time.sleep and file modification time

# --- 1. Bayesian Updater Class ---
# This class handles the sequential Bayesian inference for a normal mean.
# We assume a normal prior for the mean and a known variance for the observations.
# This setup allows for analytical updates of the posterior distribution.
class BayesianNormalMeanUpdater:
    """
    Performs sequential Bayesian inference for the mean of a Normal distribution.
    Assumes a Normal prior for the mean and a known observation variance.
    """
    def __init__(self, prior_mean: float, prior_variance: float, observation_variance: float):
        """
        Initializes the Bayesian updater with prior beliefs and observation properties.

        Args:
            prior_mean (float): The initial mean of the prior distribution for the unknown mean.
            prior_variance (float): The initial variance of the prior distribution for the unknown mean.
            observation_variance (float): The known variance of the observations (sigma^2).
        """
        if prior_variance <= 0 or observation_variance <= 0:
            raise ValueError("Variances must be positive.")

        self.mu_n = prior_mean  # Current posterior mean (mu_n)
        self.tau_n_sq = prior_variance  # Current posterior variance (tau_n^2)
        self.sigma_sq = observation_variance  # Known observation variance

        # Store initial prior for reference
        self.initial_prior_mean = prior_mean
        self.initial_prior_variance = prior_variance

        print(f"Bayesian Updater Initialized:")
        print(f"  Prior Mean (μ₀): {self.mu_n:.3f}")
        print(f"  Prior Variance (τ₀²): {self.tau_n_sq:.3f}")
        print(f"  Observation Variance (σ²): {self.sigma_sq:.3f}\n")

    def update(self, observation: float):
        """
        Updates the posterior distribution of the mean given a new observation.

        The update rules for a Normal-Normal conjugate prior are:
        New Posterior Mean (μ_n+1):
            μ_n+1 = ( (μ_n / τ_n²) + (x_n+1 / σ²) ) / ( (1 / τ_n²) + (1 / σ²) )
        New Posterior Variance (τ_n+1²):
            τ_n+1² = 1 / ( (1 / τ_n²) + (1 / σ²) )

        Args:
            observation (float): The new data point observed from the holographic file.
        """
        # Calculate precision for current posterior and observation
        precision_current_posterior = 1 / self.tau_n_sq
        precision_observation = 1 / self.sigma_sq

        # Update posterior variance
        self.tau_n_sq = 1 / (precision_current_posterior + precision_observation)

        # Update posterior mean
        self.mu_n = self.tau_n_sq * (precision_current_posterior * self.mu_n + precision_observation * observation)

    def get_posterior(self) -> tuple[float, float]:
        """
        Returns the current posterior mean and variance.

        Returns:
            tuple[float, float]: A tuple containing (posterior_mean, posterior_variance).
        """
        return self.mu_n, self.tau_n_sq

    def plot_posterior(self, ax, label: str = "Posterior"):
        """
        Plots the current posterior distribution.

        Args:
            ax (matplotlib.axes.Axes): The axes object to plot on.
            label (str): Label for the plotted distribution.
        """
        x = np.linspace(self.mu_n - 4 * np.sqrt(self.tau_n_sq), self.mu_n + 4 * np.sqrt(self.tau_n_sq), 500)
        pdf = norm.pdf(x, self.mu_n, np.sqrt(self.tau_n_sq))
        ax.plot(x, pdf, label=label)
        ax.fill_between(x, pdf, alpha=0.2)

# --- Helper function to create a sample holo.npy file ---
def create_sample_holo_file(
    filename: str = "holo.npy",
    num_points: int = 200,
    initial_mean: float = 10.0,
    change_point: int = 100,
    new_mean: float = 15.0,
    observation_std_dev: float = 2.0
):
    """
    Creates a sample .npy file with simulated holographic data.
    This file will have a change in mean at the specified change_point.
    """
    if change_point < 0 or change_point >= num_points:
        raise ValueError("Change point must be within the range [0, num_points-1].")
    if observation_std_dev <= 0:
        raise ValueError("Observation standard deviation must be positive.")

    data = np.zeros(num_points)
    data[:change_point] = np.random.normal(initial_mean, observation_std_dev, change_point)
    data[change_point:] = np.random.normal(new_mean, observation_std_dev, num_points - change_point)

    np.save(filename, data)
    print(f"Sample holographic data saved to '{filename}' with a change at index {change_point}.")
    print(f"Initial mean: {initial_mean}, New mean: {new_mean}, Std Dev: {observation_std_dev}")

# --- 3. Main Program Execution ---
def run_bayesian_hologram_analysis_from_file(filename: str = "holo.npy", ax1=None, ax2=None):
    """
    Main function to load data from a file, perform Bayesian inference, and visualize results.
    Accepts existing axes for continuous plotting.
    """
    # Bayesian Inference Parameters (Prior Beliefs)
    # These are your initial assumptions about the holographic data BEFORE observing any of it.
    prior_mean = 9.0
    prior_variance = 5.0 # A larger variance means less certainty in the prior
    # We assume a known observation variance. In a real scenario, you might estimate this
    # or use a more complex Bayesian model that infers it.
    assumed_observation_std_dev = 2.0
    assumed_observation_variance = assumed_observation_std_dev**2

    print(f"--- Starting Bayesian Holographic File Analysis for '{filename}' ---")
    print(f"Assumed observation standard deviation: {assumed_observation_std_dev}\n")

    try:
        # Load the holographic data from the specified file
        # Added .flatten() to ensure the data is a 1D array, preventing potential plotting issues.
        holographic_data = np.load(filename).flatten()
        num_observations = len(holographic_data)
        print(f"Successfully loaded {num_observations} observations from '{filename}'.\n")
        if num_observations == 0:
            print("Warning: The loaded file contains no data. Please ensure 'holo.npy' has content.")
            return

    except FileNotFoundError:
        print(f"Error: '{filename}' not found.")
        print("Please create the file first. You can run 'create_sample_holo_file()' to generate a sample.")
        return
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return

    # Initialize the Bayesian updater with our prior beliefs
    updater = BayesianNormalMeanUpdater(prior_mean, prior_variance, assumed_observation_variance)

    # Lists to store results for plotting and change detection
    posterior_means = []
    posterior_stds = []
    detected_changes = [] # Store indices where a change is detected

    # Clear previous plots if axes are provided
    if ax1 and ax2:
        ax1.clear()
        ax2.clear()
        # Re-set titles and labels as clearing also removes them
        ax1.set_title('Holographic Observations and Inferred Mean')
        ax1.set_ylabel('Observation Value')
        ax2.set_title('Evolution of Posterior Mean and Uncertainty')
        ax2.set_xlabel('Observation Number')
        ax2.set_ylabel('Posterior Mean / Std Dev')
        ax1.legend().remove() # Remove old legend to prevent duplicates
        ax2.legend().remove() # Remove old legend to prevent duplicates


    # Plot 1: Observed Data and Inferred Mean
    ax1.plot(holographic_data, 'o', markersize=3, alpha=0.6, label='Holographic Observation')
    ax1.legend() # Re-add legend after clearing

    # --- Process each observation sequentially ---
    print("\n--- Processing Observations ---")
    # Threshold for detecting a significant change in the posterior mean
    # This is a heuristic: if the mean shifts by more than X times the current std dev, flag it.
    change_detection_threshold_std_devs = 1.5
    min_observations_for_change_detection = 5 # Don't try to detect changes on very few initial points

    for i, obs in enumerate(holographic_data):
        updater.update(obs)
        current_mean, current_variance = updater.get_posterior()
        current_std = np.sqrt(current_variance)

        posterior_means.append(current_mean)
        posterior_stds.append(current_std)

        # Dynamic Change Detection Logic
        if i >= min_observations_for_change_detection:
            # Compare current mean to the mean from a few steps back (e.g., 5 steps)
            # This helps smooth out noise and detect sustained shifts.
            comparison_index = max(0, i - 5)
            mean_for_comparison = posterior_means[comparison_index]
            std_for_comparison = posterior_stds[comparison_index] # Use std from comparison point for context

            # Calculate the difference relative to the uncertainty
            mean_diff = abs(current_mean - mean_for_comparison)
            # Use a combined standard deviation for a more robust comparison
            # (e.g., sum of variances, then sqrt, or just current_std if it's stable)
            effective_std = np.mean(posterior_stds[max(0, i-5):i+1]) # Average std over a window

            if effective_std > 0 and mean_diff / effective_std > change_detection_threshold_std_devs:
                if not detected_changes or (i - detected_changes[-1]) > 10: # Avoid flagging too many consecutive points
                    print(f"\n--- Detected Potential Change at Observation {i + 1}! ---")
                    print(f"  Inferred Mean shifted from {mean_for_comparison:.3f} to {current_mean:.3f}.")
                    print(f"  This is {(mean_diff / effective_std):.2f} standard deviations of change.")
                    detected_changes.append(i)
                    # You could trigger further analysis or alerts here

        if (i + 1) % 20 == 0 or i == 0 or i == num_observations - 1:
            # Ensure obs is a float for formatting
            print(f"Observation {i + 1:3d}: Value = {float(obs):.2f} | Posterior Mean = {current_mean:.3f} ± {current_std:.3f}")

    # --- Final Plotting ---
    x_axis = np.arange(1, num_observations + 1)
    ax1.plot(x_axis - 1, posterior_means, color='green', linewidth=2, label='Inferred Posterior Mean')
    ax1.fill_between(x_axis - 1, np.array(posterior_means) - 2 * np.array(posterior_stds),
                     np.array(posterior_means) + 2 * np.array(posterior_stds),
                     color='green', alpha=0.1, label='95% Credible Interval')
    ax1.legend()

    ax2.plot(x_axis, posterior_means, color='blue', label='Posterior Mean')
    ax2.fill_between(x_axis, np.array(posterior_means) - np.array(posterior_stds),
                     np.array(posterior_means) + np.array(posterior_stds),
                     color='blue', alpha=0.2, label='±1 Posterior Std Dev')
    for change_idx in detected_changes:
        ax2.axvline(x=change_idx + 1, color='orange', linestyle=':', linewidth=2, label='Detected Change' if change_idx == detected_changes[0] else "")
    ax2.legend()

    # No plt.show() here, as we are in interactive mode
    # plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # This might cause issues with continuous updates, better to set once

    # --- Drawing Final Conclusions ---
    print("\n--- Final Conclusions Based on Bayesian Inference ---")
    final_mean, final_variance = updater.get_posterior()
    final_std = np.sqrt(final_variance)
    print(f"After processing all {num_observations} observations from '{filename}':")
    print(f"  Final Inferred Mean: {final_mean:.3f} (± {final_std:.3f})")
    print(f"  Final Posterior Variance: {final_variance:.3f}")

    if detected_changes:
        print(f"\nSummary of Detected Changes:")
        for idx in detected_changes:
            print(f"  - Significant shift detected around Observation {idx + 1}.")
        print("\nConclusion: One or more significant changes were detected in the holographic file.")
        print("Review the plots and console output for details on when these shifts occurred.")
    else:
        print("\nConclusion: No significant changes were detected in the holographic file based on the analysis.")

    print("\n--- Analysis Complete ---")

# --- Run the analysis ---
if __name__ == "__main__":
    filename = "holo.npy"
    last_modified_time = None

    # Turn on interactive plotting mode
    plt.ion()
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'Bayesian Inference for Dynamic Holographic Data from {filename}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Set layout once

    # First, create a sample holo.npy file if it doesn't exist
    if not os.path.exists(filename):
        print("Creating a sample 'holo.npy' file for demonstration...")
        create_sample_holo_file(filename)
        print("\nSample file created. Starting continuous analysis...\n")
    else:
        print(f"Using existing '{filename}' for analysis. Starting continuous monitoring...\n")
        last_modified_time = os.path.getmtime(filename)

    try:
        while True:
            current_modified_time = None
            if os.path.exists(filename):
                current_modified_time = os.path.getmtime(filename)

            if current_modified_time is None:
                print(f"Warning: '{filename}' disappeared. Waiting for it to reappear...")
                last_modified_time = None # Reset last modified time
            elif last_modified_time is None or current_modified_time > last_modified_time:
                print(f"\n--- '{filename}' modified or newly created. Re-running analysis... ---")
                # Pass the existing axes to the analysis function
                run_bayesian_hologram_analysis_from_file(filename, ax1=ax1, ax2=ax2)
                last_modified_time = current_modified_time
                fig.canvas.draw_idle() # Request a redraw
                fig.canvas.flush_events() # Process events
            else:
                print(f"'{filename}' unchanged. Checking again in 5 seconds...")

            plt.pause(0.1) # Short pause to allow plot updates and event processing
            time.sleep(4.9) # Sleep for the remainder of the 5 seconds

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
    except Exception as e:
        print(f"An unexpected error occurred during monitoring: {e}")
    finally:
        plt.ioff() # Turn off interactive mode
        plt.show() # Keep the final plot window open
