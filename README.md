---
# Conditional-WGAN for Probabilistic Regression in Energy Consumption Forecasting

## Overview

This repository contains various implementations of **Conditional Wasserstein Generative Adversarial Networks (CWGAN)** applied to **probabilistic regression** for energy consumption forecasting. The models utilize time-based and random splits for training and testing to predict future energy consumption patterns. The primary goal is to model the uncertainty in energy consumption forecasts and generate realistic probabilistic predictions.

## Contents

* **[Time-based Splits CWGAN Notebook](https://github.com/jowie1111/Conditional-WGAN-CWGAN-For-Probabilistic-Regression-in-Energy-Consumption-Forecasting/blob/main/Time%20based%20spilts%20CWGAN.ipynb)**: A Jupyter notebook demonstrating the application of CWGAN on time-based splits for forecasting energy consumption, with a focus on evaluating metrics and model performance over time.

* **[WGAN Energy Output Analysis Python Module](https://github.com/jowie1111/Conditional-WGAN-CWGAN-For-Probabilistic-Regression-in-Energy-Consumption-Forecasting/blob/main/wgan_energy_output_analysis.py)**: Python code to analyze the output of WGAN models, evaluating performance based on energy consumption forecasting and comparing predictions.

* **[Probability Regression Models Notebook](https://github.com/jowie1111/Conditional-WGAN-CWGAN-For-Probabilistic-Regression-in-Energy-Consumption-Forecasting/blob/main/Pobability%20Regression%20Models.ipynb)**: A Jupyter notebook that explores probabilistic regression models applied to energy consumption data, evaluating multiple regression approaches for better forecasting accuracy.

* **[Random Splits CWGAN and Probability Regression Model Notebook](https://github.com/jowie1111/Conditional-WGAN-CWGAN-For-Probabilistic-Regression-in-Energy-Consumption-Forecasting/blob/main/Random%20spilts%20CWGAN%20and%20%20Probability%20Regression%20Model.ipynb)**: A Jupyter notebook demonstrating the use of random splits for training CWGAN models in combination with probability regression models for improved accuracy in energy consumption forecasting.

## Requirements

To run the notebooks and Python modules, the following packages are required:

* Python 3.x
* TensorFlow (for deep learning model training)
* Keras (for defining and training models)
* NumPy (for mathematical operations)
* Pandas (for data manipulation)
* Matplotlib (for visualization)
* Scikit-learn (for machine learning algorithms and metrics)

You can install these dependencies using `pip`:

```bash
pip install tensorflow keras numpy pandas matplotlib scikit-learn
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/jowie1111/Conditional-WGAN-CWGAN-For-Probabilistic-Regression-in-Energy-Consumption-Forecasting.git
```

2. Navigate to the relevant notebook directory:

```bash
cd Conditional-WGAN-CWGAN-For-Probabilistic-Regression-in-Energy-Consumption-Forecasting
```

3. Open the Jupyter notebooks (`.ipynb` files) in Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

4. Run the code in the notebooks to train the models and evaluate the performance using different regression techniques.

## Results

The notebooks and Python modules generate various results, including:

* Time-based and random split forecasts for energy consumption.
* Evaluation of the performance of CWGAN models using different metrics.
* Probabilistic regression outputs for comparing model accuracy and uncertainty.

