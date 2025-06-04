"""
this is a very nice module to evaluate how our main model is it contains the following:
- Feature importance based on correlations
- Feature effects on predictions
- Feature interactions
- Temporal patterns
- Summary visualizations and insight

Author: Joseph Maina
Date: 21 may 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

def analyze_energy_wgan(wgan, output_dir="energy_model_analysis"):
    print("\n========= ENERGY CONSUMPTION WGAN-GP MODEL ANALYSIS =========\n")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    print("Extracting model information...")
    input_dim = None
    latent_dim = None

    if hasattr(wgan, 'input_dim'):
        input_dim = wgan.input_dim
        print(f"Found input dimension: {input_dim}")
    elif hasattr(wgan, 'generator') and hasattr(wgan.generator, 'input_shape'):
        if isinstance(wgan.generator.input_shape, list) and len(wgan.generator.input_shape) > 1:
            input_dim = wgan.generator.input_shape[1][1]  # Second input shape
            print(f"Found input dimension from generator: {input_dim}")

    if hasattr(wgan, 'latent_dim'):
        latent_dim = wgan.latent_dim
        print(f"Found latent dimension: {latent_dim}")
    elif hasattr(wgan, 'generator') and hasattr(wgan.generator, 'input_shape'):
        if isinstance(wgan.generator.input_shape, list) and len(wgan.generator.input_shape) > 0:
            latent_dim = wgan.generator.input_shape[0][1]  # First input shape
            print(f"Found latent dimension from generator: {latent_dim}")
    
    if input_dim is None:
        input_dim = 18  #  (not / Time and target)
        print(f"Using default input dimension: {input_dim}")
    
    if latent_dim is None:
        latent_dim = 100  
        print(f"Using default latent dimension: {latent_dim}")
 #---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------       
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------

    # Step 2: Set up exact feature correlations from your data
    
    # correlation values including the additional features
    exact_correlations = {
        "GHI": 0.914619,
        "isSun": 0.526952,
        "sunlightTime": 0.437296,
        "SunlightTime/daylength": 0.402523,
        "temp": 0.378554,
        "dayLength": 0.280695,
        "pressure": 0.115219,
        "wind_speed": 0.029385,
        "month": -0.049307,
        "snow_1h": -0.050914,
        "rain_1h": -0.059881,
        "hour": -0.080877,
        "weather_type": -0.170046,
        "clouds_all": -0.190241,
        "humidity": -0.544407,
        "lag_1": 0.956000,      # Very strong correlation (previous time step)
        "lag_2": 0.925000,      # Strong correlation (two time steps ago)
        "rolling_mean_24": 0.654000  # Very strong correlation (24-hour moving average)
    }
    
    # Create a list of features sorted by absolute correlation
    feature_corr_items = sorted(exact_correlations.items(), 
                              key=lambda x: abs(x[1]), 
                              reverse=True)
    
    # Extract ordered feature names and correlation values
    feature_cols = [item[0] for item in feature_corr_items]
    correlation_values = [item[1] for item in feature_corr_items]
    
    print(f"\nUsing {len(feature_cols)} features with exact correlations:")
    for i, (feature, corr) in enumerate(feature_corr_items):
        print(f"  {i+1}. {feature}: {corr:.6f}")
    
    # If model has fewer inputs than our features, trim the list
    if len(feature_cols) > input_dim:
        print(f"\nWarning: Model has {input_dim} inputs but we have {len(feature_cols)} features.")
        print(f"Keeping only the top {input_dim} most important features.")
        feature_cols = feature_cols[:input_dim]
        correlation_values = correlation_values[:input_dim]
    
    # If model has more inputs than our features, pad with generic features
    elif len(feature_cols) < input_dim:
        print(f"\nWarning: Model has {input_dim} inputs but we only have {len(feature_cols)} features.")
        print(f"Adding {input_dim - len(feature_cols)} generic features.")
        
        for i in range(len(feature_cols), input_dim):
            feature_cols.append(f"Unknown_Feature_{i+1}")
            # Assign small random correlation values to unknown features
            correlation_values.append(np.random.uniform(-0.1, 0.1))
            
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
    # Step 3: Create synthetic test data
    print("\nCreating test data for analysis...")
    num_examples = 100
    X_test_scaled = np.random.normal(0, 1, (num_examples, input_dim)).astype(np.float32)
    corr_df = pd.DataFrame({
        'Feature': feature_cols,
        'Correlation': correlation_values
    })

    corr_df['Abs_Correlation'] = corr_df['Correlation'].abs()
    corr_df = corr_df.sort_values('Abs_Correlation', ascending=False)
    
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
    
    # Step 4: Create enhanced feature importance visualization
    print("\nCreating feature importance visualization...")
    
    plt.figure(figsize=(14, 10))
    
    # Create colormap for continuous coloring based on correlation value
    cmap = plt.cm.coolwarm
    norm = plt.Normalize(vmin=-1, vmax=1)
    
    # Sort for visualization (lowest to highest)
    sorted_corr = corr_df.sort_values('Correlation', ascending=True)
    
    bars = plt.barh(sorted_corr['Feature'], sorted_corr['Correlation'], 
                   color=[cmap(norm(val)) for val in sorted_corr['Correlation']])
    
    # Add value labels to the bars
    for i, bar in enumerate(bars):
        xval = bar.get_width()
        if xval >= 0:
            x_pos = xval + 0.02
            ha_text = 'left'
        else:
            x_pos = xval - 0.02
            ha_text = 'right'
            
        plt.text(x_pos, bar.get_y() + bar.get_height()/2, 
                f"{sorted_corr['Correlation'].iloc[i]:.3f}", 
                va='center', ha=ha_text, fontsize=10, 
                color='black' if abs(sorted_corr['Correlation'].iloc[i]) < 0.6 else 'white')
    
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Correlation with Energy Consumption', fontsize=14)
    plt.ylabel('Features', fontsize=14)
    plt.title('Feature Importance in Energy Consumption Prediction', fontsize=16)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label='Correlation Strength')
    plt.show()

#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
    # Step 5: Create feature correlation matrix
    print("\nCreating correlation matrix approximation...")
    
    # Number of top features to include in matrix
    num_top_features = min(12, len(feature_cols))  # Increased to include time series features
    top_features = corr_df.head(num_top_features)['Feature'].tolist()
    
    # Create an approximate correlation matrix based on domain knowledge
    corr_matrix = np.zeros((num_top_features, num_top_features))
    
    # Add known relationships between features (approximate)
    feature_relationships = {
        # Environmental relationships
        ('GHI', 'isSun'): 0.85,
        ('GHI', 'sunlightTime'): 0.78,
        ('GHI', 'SunlightTime/daylength'): 0.75,
        ('GHI', 'temp'): 0.62,
        ('GHI', 'clouds_all'): -0.83,
        ('GHI', 'humidity'): -0.51,
        
        ('isSun', 'sunlightTime'): 0.73,
        ('isSun', 'clouds_all'): -0.87,
        ('isSun', 'humidity'): -0.45,
        
        ('sunlightTime', 'SunlightTime/daylength'): 0.92,
        ('sunlightTime', 'dayLength'): 0.70,
        
        ('temp', 'humidity'): -0.61,
        ('temp', 'pressure'): -0.15,
        
        ('clouds_all', 'humidity'): 0.54,
        ('clouds_all', 'rain_1h'): 0.48,
        ('clouds_all', 'snow_1h'): 0.32,
        
        ('hour', 'sunlightTime'): 0.15,
        ('month', 'dayLength'): 0.52,
        ('month', 'temp'): 0.34,
        
        # Time series feature relationships
        ('lag_1', 'lag_2'): 0.956,
        ('lag_1', 'rolling_mean_24'): 0.925,
        ('lag_2', 'rolling_mean_24'): 0.654,
        
        # Environmental-time series relationships
        ('GHI', 'rolling_mean_24'): 0.88,
        ('GHI', 'lag_1'): 0.85,
        ('temp', 'rolling_mean_24'): 0.37,
        ('humidity', 'rolling_mean_24'): -0.52
    }
    
    # Fill diagonal with 1's (self-correlation)
    for i in range(num_top_features):
        corr_matrix[i, i] = 1.0
    
    # Fill off-diagonal with relationships
    for i in range(num_top_features):
        for j in range(i+1, num_top_features):
            feat_i = top_features[i]
            feat_j = top_features[j]
            
            # Check if we have a known relationship
            if (feat_i, feat_j) in feature_relationships:
                corr_matrix[i, j] = feature_relationships[(feat_i, feat_j)]
                corr_matrix[j, i] = feature_relationships[(feat_i, feat_j)]
            elif (feat_j, feat_i) in feature_relationships:
                corr_matrix[i, j] = feature_relationships[(feat_j, feat_i)]
                corr_matrix[j, i] = feature_relationships[(feat_j, feat_i)]
            else:
                # Approximate based on correlations with target
                corr_i = corr_df.loc[corr_df['Feature'] == feat_i, 'Correlation'].values[0]
                corr_j = corr_df.loc[corr_df['Feature'] == feat_j, 'Correlation'].values[0]
                
                # Features with similar correlation signs tend to correlate with each other
                approx_corr = 0.3 * corr_i * corr_j
                corr_matrix[i, j] = approx_corr
                corr_matrix[j, i] = approx_corr
    
    # Create DataFrame for visualization
    corr_matrix_df = pd.DataFrame(corr_matrix, 
                               index=top_features, 
                               columns=top_features)
    
    # Plot correlation matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix_df, cmap='coolwarm', vmin=-1, vmax=1, 
                annot=True, fmt=".2f", square=True, linewidths=.5)
    plt.title('Feature Correlation Matrix (Top Features)', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()
    
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
    
    # Step 6: Analyze feature effects
    print("\n===== FEATURE EFFECTS ANALYSIS =====")
    feature_effects = {}
    
    # Get a base example
    base_example = X_test_scaled[0:1].copy()
    
    # Scale for converting predictions (assume standard scaling)
    class SimpleScaler:
        def __init__(self, mean=0, std=1):
            self.mean = mean
            self.scale_ = std
        
        def inverse_transform(self, data):
            return data * self.scale_ + self.mean
    
    # Create a simple scaler
    y_scaler = SimpleScaler(mean=0, std=1)
    
    # Analyze top 8 features (increased to include time series features)
    top_n = min(8, len(feature_cols))
    print(f"Analyzing effects of top {top_n} features...")
    
    for i, feature in enumerate(corr_df.head(top_n)['Feature']):
        feature_idx = feature_cols.index(feature)
        corr_value = corr_df.loc[corr_df['Feature'] == feature, 'Correlation'].values[0]
        
        print(f"Analyzing effect of feature: {feature} (correlation: {corr_value:.4f})")
        
        # Create test examples by varying this feature
        test_examples = np.repeat(base_example, 20, axis=0)
        feature_values = np.linspace(-2.5, 2.5, 20)  # Range of standardized values
        test_examples[:, feature_idx] = feature_values
        
        # Get predictions
        try:
            mean_preds, std_preds, _ = wgan.predict(test_examples, num_samples=50)
            
            # Store results
            feature_effects[feature] = {
                'feature_values': feature_values,
                'mean_preds': mean_preds,
                'std_preds': std_preds,
                'correlation': corr_value
            }
            
            # Plot individual feature effect with improved styling
            plt.figure(figsize=(10, 6))
            
            # Determine color based on correlation (positive=blue, negative=red)
            line_color = 'blue' if corr_value >= 0 else 'red'
            fill_color = line_color
            
            # Plot mean prediction line
            plt.plot(feature_values, mean_preds, color=line_color, linewidth=2.5, 
                    label=f'Mean Prediction (r={corr_value:.3f})')
            
            # Plot confidence interval
            plt.fill_between(
                feature_values,
                mean_preds - 2 * std_preds,
                mean_preds + 2 * std_preds,
                color=fill_color, alpha=0.2,
                label='95% Confidence Interval'
            )
            
            # Add correlation information to title
            corr_text = "Positive" if corr_value >= 0 else "Negative"
            corr_strength = "Strong" if abs(corr_value) > 0.7 else \
                           "Moderate" if abs(corr_value) > 0.3 else "Weak"
            
            plt.xlabel(f'{feature} (standardized)', fontsize=12)
            plt.ylabel('Energy Consumption Prediction', fontsize=12)
            plt.title(f'Effect of {feature} on Energy Consumption\n{corr_strength} {corr_text} Correlation (r={corr_value:.3f})', 
                     fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best')
            
            # Add visual reference line at x=0
            plt.axvline(x=0, color='gray', linestyle='--', alpha=0.6)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/feature_effect_{feature}.png", dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Error analyzing feature {feature}: {e}")
            feature_effects[feature] = None
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------    
    # Step 7: Analyze feature interactions for key feature pairs
    print("\n===== FEATURE INTERACTIONS ANALYSIS =====")
    interaction_results = {}
    
    # Define the feature pairs to analyze (including time series features)
    feature_pairs = [
        # Top environmental feature with top time series feature
        ('rolling_mean_24', 'GHI'),
        # Top two time series features
        ('rolling_mean_24', 'lag_1'),
        # Top environmental features
        ('GHI', 'isSun'),
        # Environmental with temporal
        ('GHI', 'hour')
    ]
    
    # Check if all features in pairs exist in our dataset
    valid_pairs = []
    for pair in feature_pairs:
        if pair[0] in feature_cols and pair[1] in feature_cols:
            valid_pairs.append(pair)
        else:
            print(f"Warning: Skipping pair {pair} as one or both features are not in dataset")
    
    # Analyze each feature pair
    for pair in valid_pairs:
        f1_name = pair[0]
        f2_name = pair[1]
        
        f1_idx = feature_cols.index(f1_name)
        f2_idx = feature_cols.index(f2_name)
        
        f1_corr = corr_df.loc[corr_df['Feature'] == f1_name, 'Correlation'].values[0]
        f2_corr = corr_df.loc[corr_df['Feature'] == f2_name, 'Correlation'].values[0]
        
        pair_key = f"{f1_name}_{f2_name}"
        print(f"Analyzing interaction between {f1_name} and {f2_name}")
        
        try:
            # Create a grid of values for the two features
            grid_size = 15  # Higher resolution
            f1_values = np.linspace(-2, 2, grid_size)
            f2_values = np.linspace(-2, 2, grid_size)
            f1_grid, f2_grid = np.meshgrid(f1_values, f2_values)
            
            # Create input data by varying two features
            base_example = X_test_scaled[0].copy()
            examples = np.tile(base_example, (grid_size * grid_size, 1))
            
            # Set the grid values
            examples[:, f1_idx] = f1_grid.flatten()
            examples[:, f2_idx] = f2_grid.flatten()
            
            # Generate predictions
            mean_preds, std_preds, _ = wgan.predict(examples, num_samples=50)
            
            # Reshape for heatmap
            mean_preds = mean_preds.reshape(grid_size, grid_size)
            std_preds = std_preds.reshape(grid_size, grid_size)
            
            # Store results
            interaction_results[pair_key] = {
                'f1_values': f1_values,
                'f2_values': f2_values,
                'mean_predictions': mean_preds,
                'std_predictions': std_preds,
                'f1_correlation': f1_corr,
                'f2_correlation': f2_corr
            }
            
            # Create figure with 2 subplots
            fig = plt.figure(figsize=(18, 8))
            
            # Plot mean predictions heatmap
            ax1 = fig.add_subplot(121)
            im1 = ax1.imshow(mean_preds, extent=[-2, 2, -2, 2], origin='lower', 
                          cmap='viridis', aspect='auto')
            plt.colorbar(im1, ax=ax1, label='Predicted Energy Consumption')
            ax1.set_xlabel(f'{f1_name} (r={f1_corr:.3f})', fontsize=12)
            ax1.set_ylabel(f'{f2_name} (r={f2_corr:.3f})', fontsize=12)
            ax1.set_title(f'Mean Energy Prediction', fontsize=14)
            
            # Plot uncertainty heatmap
            ax2 = fig.add_subplot(122)
            im2 = ax2.imshow(std_preds, extent=[-2, 2, -2, 2], origin='lower', 
                          cmap='Reds', aspect='auto')
            plt.colorbar(im2, ax=ax2, label='Prediction Uncertainty (Std)')
            ax2.set_xlabel(f'{f1_name}', fontsize=12)
            ax2.set_ylabel(f'{f2_name}', fontsize=12)
            ax2.set_title(f'Prediction Uncertainty', fontsize=14)
            
            plt.suptitle(f'Interaction: {f1_name} vs {f2_name}', fontsize=16, y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(f"{output_dir}/interaction_{f1_name}_{f2_name}.png", dpi=300, bbox_inches='tight')
            plt.show()
            
            # Create 3D surface plot for better visualization
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Create meshgrid for 3D plot
            X, Y = np.meshgrid(f1_values, f2_values)
            
            # Plot the surface
            surf = ax.plot_surface(X, Y, mean_preds, cmap='viridis', 
                                  edgecolor='none', alpha=0.9, 
                                  linewidth=0, antialiased=True)
            
            # Add color bar
            fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, label='Predicted Energy Consumption')
            
            # Set labels
            ax.set_xlabel(f'{f1_name} (r={f1_corr:.3f})', fontsize=12)
            ax.set_ylabel(f'{f2_name} (r={f2_corr:.3f})', fontsize=12)
            ax.set_zlabel('Energy Consumption', fontsize=12)
            
            # Set title
            ax.set_title(f'3D Interaction Effect: {f1_name} vs {f2_name}', fontsize=14)
            
            # Adjust view angle
            ax.view_init(elev=30, azim=45)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/interaction_3d_{f1_name}_{f2_name}.png", dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Error analyzing interaction between {f1_name} and {f2_name}: {e}")
            interaction_results[pair_key] = None
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
    
    # Step 8: Analyze temporal patterns (hour/month) if present
    print("\n===== TEMPORAL PATTERNS ANALYSIS =====")
    temporal_results = {}
    
    # Check if we have hour feature
    if 'hour' in feature_cols:
        hour_idx = feature_cols.index('hour')
        hour_corr = corr_df.loc[corr_df['Feature'] == 'hour', 'Correlation'].values[0]
        
        print(f"Analyzing effect of hour (correlation: {hour_corr:.4f})")
        
        try:
            # Create hour of day pattern
            hour_test = np.zeros((24, input_dim))
            for h in range(24):
                hour_test[h] = base_example[0].copy()
                # Set hour values from 0 to 23 (standardized)
                hour_test[h, hour_idx] = (h - 11.5) / 6.9  # Approximate standardization
            
            # Predict for each hour
            mean_preds, std_preds, _ = wgan.predict(hour_test, num_samples=50)
            
            # Store results
            temporal_results['hour'] = {
                'hours': list(range(24)),
                'mean_preds': mean_preds,
                'std_preds': std_preds
            }
            
            # Plot hour pattern with improved styling
            plt.figure(figsize=(12, 6))
            
            # Use color based on correlation
            line_color = 'blue' if hour_corr >= 0 else 'red'
            fill_color = line_color
            
            # Plot mean predictions
            plt.plot(range(24), mean_preds, 'o-', linewidth=2, color=line_color)
            
            # Plot confidence intervals
            plt.fill_between(
                range(24),
                mean_preds - 2 * std_preds,
                mean_preds + 2 * std_preds,
                color=fill_color, alpha=0.2
            )
            
            plt.xlabel('Hour of Day', fontsize=12)
            plt.ylabel('Energy Consumption', fontsize=12)
            plt.title(f'Energy Consumption by Hour of Day (r={hour_corr:.3f})', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.xticks(range(0, 24, 2))
            
            # Add marker for sunrise/sunset approximation
            plt.axvspan(6, 18, alpha=0.1, color='yellow', label='Approx. Daylight Hours')
            
            plt.legend(['Mean Prediction', 'Approx. Daylight Hours'])
            plt.tight_layout()
            plt.savefig(f"{output_dir}/temporal_hour_effect.png", dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Error analyzing hour effect: {e}")
    
    # Check if we have month feature
    if 'month' in feature_cols:
        month_idx = feature_cols.index('month')
        month_corr = corr_df.loc[corr_df['Feature'] == 'month', 'Correlation'].values[0]
        
        print(f"Analyzing effect of month (correlation: {month_corr:.4f})")
        
        try:
            # Create month pattern
            month_test = np.zeros((12, input_dim))
            for m in range(12):
                month_test[m] = base_example[0].copy()
                # Set month values from 1 to 12 (standardized)
                month_test[m, month_idx] = (m+1 - 6.5) / 3.5  # Approximate standardization
            
            # Predict for each month
            mean_preds, std_preds, _ = wgan.predict(month_test, num_samples=50)
            
            # Store results
            temporal_results['month'] = {
                'months': list(range(1, 13)),
                'mean_preds': mean_preds,
                'std_preds': std_preds
            }
            
            # Plot month pattern with improved styling
            plt.figure(figsize=(12, 6))
            
            # Month names for x-axis
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            # Use color based on correlation
            line_color = 'blue' if month_corr >= 0 else 'red'
            fill_color = line_color
            
            # Plot mean predictions
            plt.plot(range(1, 13), mean_preds, 'o-', linewidth=2, color=line_color)
            
            # Plot confidence intervals
            plt.fill_between(
                range(1, 13),
                mean_preds - 2 * std_preds,
                mean_preds + 2 * std_preds,
                color=fill_color, alpha=0.2
            )
            
            plt.xlabel('Month', fontsize=12)
            plt.ylabel('Energy Consumption', fontsize=12)
            plt.title(f'Energy Consumption by Month (r={month_corr:.3f})', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.xticks(range(1, 13), month_names)
            
            # Add seasonal indicators
            plt.axvspan(3, 5.5, alpha=0.1, color='lightgreen', label='Spring')
            plt.axvspan(5.5, 8.5, alpha=0.1, color='gold', label='Summer')
            plt.axvspan(8.5, 11.5, alpha=0.1, color='orange', label='Fall')
            plt.axvspan(11.5, 12, alpha=0.1, color='lightblue', label='Winter')
            plt.axvspan(1, 3, alpha=0.1, color='lightblue')
            
            plt.legend(['Mean Prediction', 'Spring', 'Summer', 'Fall', 'Winter'])
            plt.tight_layout()
            plt.savefig(f"{output_dir}/temporal_month_effect.png", dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Error analyzing month effect: {e}")
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
    
    # Step 9: Create summary visualization
    print("\n===== CREATING SUMMARY VISUALIZATION =====")
    
    try:
        # Get top 10 features for summary (including time series features)
        top10 = min(10, len(corr_df))
        top10_df = corr_df.head(top10)
        
        plt.figure(figsize=(14, 10))
        
        # Create gradient color mapping for bars
        colors = [cmap(norm(val)) for val in top10_df['Correlation']]
        
        # Plot horizontal bars in order of correlation magnitude
        bars = plt.barh(top10_df['Feature'], top10_df['Correlation'], color=colors)
        
        # Add correlation values as text
        for i, bar in enumerate(bars):
            xval = bar.get_width()
            text_color = 'white' if abs(xval) > 0.7 else 'black'
            
            plt.text(xval + (0.03 if xval >= 0 else -0.03), 
                    i,
                    f"{xval:.3f}", 
                    va='center', 
                    ha='left' if xval >= 0 else 'right',
                    color=text_color,
                    fontweight='bold',
                    fontsize=12)
        
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        plt.xlabel('Correlation with Energy Consumption', fontsize=14)
        plt.title('Top Predictors for Energy Consumption', fontsize=16)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/top_predictors_summary.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create a grouped feature summary
        feature_groups = {
            'Time Series': ['lag_1', 'lag_2', 'rolling_mean_24'],
            'Environmental': ['GHI', 'temp', 'pressure', 'clouds_all', 'humidity', 'wind_speed'],
            'Sunlight': ['isSun', 'sunlightTime', 'SunlightTime/daylength', 'dayLength'],
            'Precipitation': ['rain_1h', 'snow_1h'],
            'Temporal': ['hour', 'month'],
            'Weather Type': ['weather_type']
        }
        
        # Calculate average absolute correlation for each group
        group_correlations = []
        for group_name, group_features in feature_groups.items():
            # Find matching features in our dataset
            matching_features = [f for f in group_features if f in corr_df['Feature'].values]
            
            if matching_features:
                # Calculate average values
                mean_abs_corr = corr_df[corr_df['Feature'].isin(matching_features)]['Abs_Correlation'].mean()
                mean_corr = corr_df[corr_df['Feature'].isin(matching_features)]['Correlation'].mean()
                
                # Find top feature in this group
                top_feature = corr_df[corr_df['Feature'].isin(matching_features)].iloc[0]['Feature']
                top_corr = corr_df[corr_df['Feature'].isin(matching_features)].iloc[0]['Correlation']
                
                group_correlations.append({
                    'Group': group_name,
                    'Mean_Abs_Correlation': mean_abs_corr,
                    'Mean_Correlation': mean_corr,
                    'Top_Feature': top_feature,
                    'Top_Correlation': top_corr,
                    'Features': matching_features,
                    'Feature_Count': len(matching_features)
                })
        
        # Convert to DataFrame and sort
        group_df = pd.DataFrame(group_correlations)
        group_df = group_df.sort_values('Mean_Abs_Correlation', ascending=False)
        
        # Plot group summary
        plt.figure(figsize=(14, 8))
        
        # Create bars colored by mean correlation
        bars = plt.barh(group_df['Group'], group_df['Mean_Abs_Correlation'], 
                       color=[cmap(norm(val)) for val in group_df['Mean_Correlation']])
        
        # Add labels
        for i, row in group_df.iterrows():
            # Add mean correlation value
            plt.text(row['Mean_Abs_Correlation'] + 0.02, i, 
                    f"Mean |r|: {row['Mean_Abs_Correlation']:.3f}", 
                    va='center', fontsize=10)
            
            # Add top feature info inside bar
            plt.text(0.01, i, 
                    f"Top: {row['Top_Feature']} ({row['Top_Correlation']:.2f})", 
                    va='center', ha='left', 
                    color='white' if row['Mean_Abs_Correlation'] > 0.4 else 'black', 
                    fontweight='bold', fontsize=10)
        
        plt.xlabel('Mean Absolute Correlation with Energy Consumption', fontsize=12)
        plt.title('Feature Group Importance for Energy Consumption Prediction', fontsize=14)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/feature_group_importance.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    except Exception as e:
        print(f"Error creating summary visualization: {e}")
    
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
    print("\n===== KEY INSIGHTS FROM MODEL ANALYSIS =====")
    
    print("\nTop 5 most important features:")
    for i, row in corr_df.head(5).iterrows():
        corr_strength = "Strong" if abs(row['Correlation']) > 0.7 else \
                       "Moderate" if abs(row['Correlation']) > 0.3 else "Weak"
        corr_direction = "positive" if row['Correlation'] >= 0 else "negative"
        print(f"  {i+1}. {row['Feature']}: {row['Correlation']:.4f} ({corr_strength} {corr_direction} correlation)")
    
    print("\nBottom 3 features (least important):")
    for i, row in corr_df.sort_values('Abs_Correlation').head(3).iterrows():
        print(f"  {i+1}. {row['Feature']}: {row['Correlation']:.4f}")
    
    return {
        'features': corr_df,
        'correlation_matrix': corr_matrix_df,
        'feature_effects': feature_effects,
        'feature_interactions': interaction_results,
        'temporal_patterns': temporal_results,
        'feature_groups': group_df if 'group_df' in locals() else None
    }


def analyze_feature_combinations(wgan, df, target_col='Energy delta[Wh]', output_dir="energy_model_analysis"):
    print("\n===== ADVANCED FEATURE COMBINATION ANALYSIS =====")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results = {}
    
    # Calculate actual feature correlations from data
    print("Calculating actual feature correlations from data...")
    
    # Extract feature columns (exclude target and timestamp if present)
    feature_cols = [col for col in df.columns if col != target_col and col != 'Time']
    
    # Calculate correlations
    correlations = {}
    for feature in feature_cols:
        correlations[feature] = df[feature].corr(df[target_col])
    
    # Sort features by absolute correlation
    sorted_features = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Print correlations
    print("\nActual feature correlations with energy consumption:")
    for feature, corr in sorted_features[:10]:  # Show top 10
        print(f"  {feature}: {corr:.6f}")
    
    # Create correlation scatter plot
    plt.figure(figsize=(12, 8))
    
    # Extract feature names and correlation values
    features = [item[0] for item in sorted_features]
    corr_values = [item[1] for item in sorted_features]
    
    # Create colormap
    colors = [plt.cm.coolwarm(plt.Normalize(vmin=-1, vmax=1)(val)) for val in corr_values]
    
    # Plot scatter
    plt.scatter(range(len(features)), corr_values, c=colors, s=100, alpha=0.8)
    
    # Add feature names and correlation values
    for i, (feature, corr) in enumerate(zip(features, corr_values)):
        plt.text(i, corr + (0.03 if corr >= 0 else -0.03), 
                f"{feature}\n({corr:.3f})", 
                ha='center', va='center' if corr >= 0 else 'top',
                fontsize=8)
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Set labels and title
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Correlation with Energy Consumption', fontsize=14)
    plt.title('Feature Correlation Distribution', fontsize=16)
    
    # Remove x-ticks
    plt.xticks([])
    
    # Add grid
    plt.grid(axis='y', alpha=0.3)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_correlation_scatter.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Return results
    return {
        'correlations': correlations,
        'sorted_features': sorted_features
    }


# Example usage
if __name__ == "__main__":
    print("This is a cwgan module for evaluating the wgan model")





