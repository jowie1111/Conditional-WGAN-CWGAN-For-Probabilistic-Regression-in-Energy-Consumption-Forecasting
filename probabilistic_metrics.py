#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

tfd = tfp.distributions

# PROBABILISTIC EVALUATION METRICS
# ===============================

def continuous_ranked_probability_score(y_true, y_pred_mean, y_pred_std):

    # Create a normal distribution for each prediction
    distributions = [tfp.distributions.Normal(loc=mu, scale=sigma) 
                    for mu, sigma in zip(y_pred_mean, y_pred_std)]
    
    crps_values = []
    for dist, actual in zip(distributions, y_true):
        # Generate CDF evaluation points centered on actual value
        eval_points = np.linspace(actual - 5 * dist.stddev(), actual + 5 * dist.stddev(), 1000)
        
        # Calculate CDF values at evaluation points
        cdf_values = dist.cdf(eval_points).numpy()
        
        # Step function for actual value (1 if x >= actual, 0 otherwise)
        step_values = np.array([1.0 if x >= actual else 0.0 for x in eval_points])
        
        # Calculate CRPS as integral of squared difference
        squared_diff = (cdf_values - step_values) ** 2
        crps = np.trapz(squared_diff, eval_points)
        crps_values.append(crps)
    
    return np.mean(crps_values)

def pinball_loss(y_true, y_pred_quantiles, quantiles):

    losses = []
    
    for q in quantiles:
        y_pred = y_pred_quantiles[q]
        errors = y_true - y_pred
        loss = np.mean(np.maximum(q * errors, (q - 1) * errors))
        losses.append(loss)
    
    return np.mean(losses)

def generate_quantile_forecasts_from_gaussian(y_pred_mean, y_pred_std, quantiles):

    quantile_forecasts = {}
    
    for q in quantiles:
        # Use the percent point function (inverse of CDF) to get quantiles
        quantile_forecasts[q] = y_pred_mean + norm.ppf(q) * y_pred_std
    
    return quantile_forecasts

def wasserstein_distance(samples_actual, samples_predicted):
 
    # Reshape to 1D arrays
    samples_actual = samples_actual.reshape(-1, 1)
    samples_predicted = samples_predicted.reshape(-1, 1)
    
    # Ensure both sets of samples have the same shape
    if len(samples_actual) > len(samples_predicted):
        samples_actual = samples_actual[:len(samples_predicted)]
    elif len(samples_predicted) > len(samples_actual):
        samples_predicted = samples_predicted[:len(samples_actual)]
    
    # Compute the cost matrix: all pairwise L1 distances
    cost_matrix = cdist(samples_actual, samples_predicted, 'cityblock')
    
    # Use a simple approximation for demonstration purposes
    # (Note: A full optimal transport solution would be more accurate)
    # Sort both samples and compute mean absolute difference
    samples_actual_sorted = np.sort(samples_actual.flatten())
    samples_predicted_sorted = np.sort(samples_predicted.flatten())
    
    return np.mean(np.abs(samples_actual_sorted - samples_predicted_sorted))

def interval_coverage(y_true, y_pred_mean, y_pred_std, confidence_level=0.95):

    # Calculate critical value for the confidence level
    alpha = 1 - confidence_level
    z_score = norm.ppf(1 - alpha/2)
    
    # Compute lower and upper bounds
    lower_bound = y_pred_mean - z_score * y_pred_std
    upper_bound = y_pred_mean + z_score * y_pred_std
    
    # Count how many actual values fall within the interval
    within_interval = np.logical_and(y_true >= lower_bound, y_true <= upper_bound)
    coverage_rate = np.mean(within_interval)
    
    return coverage_rate

def energy_score(y_true, y_pred_samples):

    n_samples = y_pred_samples.shape[0]
    term1 = 0
    
    # Calculate first term: average Euclidean distance between 
    # each ensemble member and the observation
    for i in range(n_samples):
        term1 += np.sqrt(np.sum((y_pred_samples[i] - y_true) ** 2))
    term1 /= n_samples
    
    # Calculate second term: average Euclidean distance between ensemble members
    term2 = 0
    for i in range(n_samples):
        for j in range(n_samples):
            term2 += np.sqrt(np.sum((y_pred_samples[i] - y_pred_samples[j]) ** 2))
    term2 /= (2 * n_samples * n_samples)
    
    # Energy score is term1 - term2
    return term1 - term2

def plot_reliability_diagram(y_true, y_pred_mean, y_pred_std, bins=10):

    # Calculate predicted probabilities for each observation
    observations = []
    predicted_probs = []
    
    # For each data point
    for actual, mean, std in zip(y_true, y_pred_mean, y_pred_std):
        # Create a normal distribution for the prediction
        dist = norm(loc=mean, scale=std)
        
        # Calculate the predicted probability that the value is less than or equal to actual
        prob = dist.cdf(actual)
        
        # Store observation (0 or 1) and predicted probability
        observations.append(actual <= mean)  # Binary indicator
        predicted_probs.append(prob)
    
    # Convert to numpy arrays
    observations = np.array(observations)
    predicted_probs = np.array(predicted_probs)
    
    # Calculate reliability curve
    bin_edges = np.linspace(0, 1, bins + 1)
    bin_indices = np.digitize(predicted_probs, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, bins - 1)
    
    bin_sums = np.bincount(bin_indices, weights=observations, minlength=bins)
    bin_counts = np.bincount(bin_indices, minlength=bins)
    bin_counts = np.where(bin_counts == 0, 1, bin_counts)  # Avoid division by zero
    bin_probs = bin_sums / bin_counts
    
    # Bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Plot reliability diagram
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.plot(bin_centers, bin_probs, 'o-', label='Model Calibration')
    
    # Add histogram of predicted probabilities (helps with interpretation)
    ax2 = plt.gca().twinx()
    ax2.hist(predicted_probs, bins=bin_edges, alpha=0.3, color='gray')
    
    plt.xlabel('Predicted Probability')
    plt.ylabel('Observed Frequency')
    ax2.set_ylabel('Count')
    plt.title('Reliability Diagram (Calibration Curve)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    return plt.gcf()

def calibrate_uncertainty(y_train, y_pred_train_mean, y_pred_train_std, y_pred_test_std):

    # Calculate absolute errors on training set
    abs_errors = np.abs(y_train - y_pred_train_mean)
    
    # Fit isotonic regression to map predicted std to observed errors
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(y_pred_train_std, abs_errors)
    
    # Calibrate test uncertainties
    calibrated_std = iso_reg.predict(y_pred_test_std)
    
    return calibrated_std

def evaluate_all_probabilistic_metrics(y_true, y_pred_mean, y_pred_std, num_samples=1000):

    # Generate ensemble samples
    ensemble_samples = np.array([
        y_pred_mean + np.random.normal(0, 1, len(y_pred_mean)) * y_pred_std
        for _ in range(num_samples)
    ])
    
    # Generate quantile forecasts
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    quantile_forecasts = generate_quantile_forecasts_from_gaussian(y_pred_mean, y_pred_std, quantiles)
    
    # Calculate metrics
    crps = continuous_ranked_probability_score(y_true, y_pred_mean, y_pred_std)
    pinball = pinball_loss(y_true, quantile_forecasts, quantiles)
    wass_dist = wasserstein_distance(y_true, ensemble_samples.flatten())
    
    # Calculate interval coverage for different confidence levels
    coverage_50 = interval_coverage(y_true, y_pred_mean, y_pred_std, 0.5)
    coverage_90 = interval_coverage(y_true, y_pred_mean, y_pred_std, 0.9)
    coverage_95 = interval_coverage(y_true, y_pred_mean, y_pred_std, 0.95)
    
    # Calculate energy score
    en_score = energy_score(y_true, ensemble_samples)
    
    return {
        'CRPS': crps,
        'Pinball Loss': pinball,
        'Wasserstein Distance': wass_dist,
        'Interval Coverage (50%)': coverage_50,
        'Interval Coverage (90%)': coverage_90, 
        'Interval Coverage (95%)': coverage_95,
        'Energy Score': en_score
    }

def generate_samples_from_wgan(wgan_model, X_test, num_samples=30):
  
    samples = []
    
    for _ in range(num_samples):
        # Generate random noise
        noise = tf.random.normal([X_test.shape[0], wgan_model.latent_dim])
        
        # Generate predictions
        pred = wgan_model.generator.predict([noise, X_test], verbose=0)
        samples.append(pred.flatten())
    
    return np.array(samples)

def compare_models_probabilistic_metrics(models_predictions, models_uncertainties, y_test):

    results = {}
    
    for model_name, y_pred in models_predictions.items():
        if model_name in models_uncertainties:
            y_std = models_uncertainties[model_name]
            
            # Evaluate metrics
            metrics = evaluate_all_probabilistic_metrics(y_test, y_pred, y_std)
            results[model_name] = metrics
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

def calculate_quantile_from_gaussian(mean, std, quantile):
    """Helper function to calculate a specific quantile from Gaussian parameters"""
    return mean + norm.ppf(quantile) * std

def plot_model_evaluation_charts(models_predictions, models_uncertainties, y_test):

    # 1. Prediction Interval Coverage
    plt.figure(figsize=(15, 10))
    
    # Set up 3 confidence levels to check
    confidence_levels = [0.5, 0.9, 0.95]
    colors = ['b', 'g', 'r', 'purple', 'orange']
    
    for i, (model_name, y_pred) in enumerate(models_predictions.items()):
        if model_name in models_uncertainties:
            y_std = models_uncertainties[model_name]
            
            # Calculate coverage for different confidence levels
            coverages = [interval_coverage(y_test, y_pred, y_std, cl) for cl in confidence_levels]
            
            # Plot as bar chart
            x = np.arange(len(confidence_levels)) + i * 0.15
            plt.bar(x, coverages, width=0.15, color=colors[i], alpha=0.7, label=model_name)
            
            # Plot ideal line for each confidence level
            for j, cl in enumerate(confidence_levels):
                plt.plot([j - 0.4, j + 0.6], [cl, cl], 'k--', alpha=0.5)
    
    plt.xlabel('Confidence Level')
    plt.ylabel('Actual Coverage Rate')
    plt.title('Prediction Interval Coverage by Model')
    plt.xticks(np.arange(len(confidence_levels)), [f'{cl*100:.0f}%' for cl in confidence_levels])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('interval_coverage_comparison.png')
    
    # 2. Plot Reliability Diagrams for each model
    for model_name, y_pred in models_predictions.items():
        if model_name in models_uncertainties:
            y_std = models_uncertainties[model_name]
            fig = plot_reliability_diagram(y_test, y_pred, y_std)
            fig.suptitle(f'Reliability Diagram for {model_name}')
            plt.tight_layout()
            plt.savefig(f'reliability_diagram_{model_name}.png')
    
    # 3. Plot predictions with uncertainty
    plt.figure(figsize=(15, 10))
    
    # Limited sample for visibility
    sample_size = 100
    indices = np.arange(min(sample_size, len(y_test)))
    
    # Plot actual values
    plt.plot(indices, y_test[:sample_size], 'k-', label='Actual', linewidth=2)
    
    # Plot predictions with uncertainty
    for i, (model_name, y_pred) in enumerate(models_predictions.items()):
        if model_name in models_uncertainties:
            y_std = models_uncertainties[model_name]
            
            # Only show first few samples for clarity
            plt.plot(indices, y_pred[:sample_size], '-', color=colors[i], label=f'{model_name} Prediction')
            
            # Plot 95% confidence interval
            plt.fill_between(
                indices,
                y_pred[:sample_size] - 1.96 * y_std[:sample_size],
                y_pred[:sample_size] + 1.96 * y_std[:sample_size],
                alpha=0.2, color=colors[i]
            )
    
    plt.xlabel('Sample Index')
    plt.ylabel('Energy Consumption')
    plt.title('Model Predictions with Uncertainty')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('model_predictions_comparison.png')
    
    # 4. Plot uncertainty distribution
    plt.figure(figsize=(15, 6))
    
    for i, (model_name, _) in enumerate(models_predictions.items()):
        if model_name in models_uncertainties:
            y_std = models_uncertainties[model_name]
            plt.hist(y_std, bins=30, alpha=0.5, label=model_name, color=colors[i])
    
    plt.xlabel('Predicted Uncertainty (Std)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predicted Uncertainties')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('uncertainty_distribution.png')

# Main execution function
def evaluate_probabilistic_models(wgan_model, X_test_scaled, y_test, y_scaler, models_predictions, models_uncertainties):

    # Scale y_test for consistent comparison
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
    
    # Generate samples from WGAN for ensemble-based metrics
    print("Generating samples from WGAN for ensemble metrics...")
    wgan_samples = generate_samples_from_wgan(wgan_model, X_test_scaled, num_samples=100)
    
    # If needed, transform samples back to original scale
    wgan_samples_orig = np.array([y_scaler.inverse_transform(sample.reshape(-1, 1)).flatten() 
                                  for sample in wgan_samples])
    
    # Calculate WGAN-specific Wasserstein distance (between actual and generated distributions)
    print("Calculating Wasserstein distance...")
    w_dist = wasserstein_distance(y_test, wgan_samples_orig.flatten())
    print(f"Wasserstein distance: {w_dist:.4f}")
    
    # Evaluate all probabilistic models with comprehensive metrics
    print("Evaluating all probabilistic models...")
    metrics_comparison = compare_models_probabilistic_metrics(
        models_predictions, models_uncertainties, y_test)
    
    print("\n======== PROBABILISTIC METRICS EVALUATION ========")
    print(metrics_comparison)
    
    # Plot comprehensive evaluation charts
    print("Generating evaluation charts...")
    plot_model_evaluation_charts(models_predictions, models_uncertainties, y_test)
    
    # Return results
    return {
        'metrics_comparison': metrics_comparison,
        'wasserstein_distance': w_dist,
        'wgan_samples': wgan_samples_orig
    }

# Function to run from main code
def run_probabilistic_evaluation(X_train, y_train, X_test, y_test, trained_models, predictions, uncertainties):

    # Ensure y_test is a numpy array
    y_test_np = y_test.to_numpy() if isinstance(y_test, pd.Series) else y_test
    
    # Scale data for WGAN
    X_scaler = StandardScaler()
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1) if isinstance(y_train, pd.Series) 
                                           else y_train.reshape(-1, 1)).flatten()
    
    # Get the WGAN model
    wgan_model = trained_models.get('WGAN')
    
    # If wgan_model doesn't exist in trained_models, you'd need to recreate it
    # For demonstration, we'll assume it exists
    
    # Run evaluation
    results = evaluate_probabilistic_models(
        wgan_model, X_test_scaled, y_test_np, y_scaler, predictions, uncertainties)
    
    # Save results
    metrics_df = results['metrics_comparison']
    metrics_df.to_csv('probabilistic_metrics_results.csv')
    
    print("\nProbabilistic evaluation complete. Results saved to CSV and PNG files.")
    return results

if __name__ == "__main__":
    print("from probabilistic_metrics import run_probabilistic_evaluation")
    print("results = run_probabilistic_evaluation(X_train, y_train, X_test, y_test, trained_models, predictions, uncertainties)")


