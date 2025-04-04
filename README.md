# Energy Consumption Forecasting for 5G Base Stations Using Machine Learning

## Overview

The operational expenditure (OPEX) for telecom operators is significantly driven by energy consumption, with over 70% of this energy used by the radio access network (RAN)—particularly the base stations (BSs). This project leverages machine learning to develop predictive models that estimate the energy consumed by 5G base stations under various conditions.

By considering factors such as engineering configurations, traffic patterns, and energy-saving techniques, the model aims to provide insights that can help optimize energy usage and reduce overall costs.

## Objectives

- Build a predictive model to estimate energy consumption of 5G base stations.
- Analyze the impact of:
  - Traffic load and network usage patterns.
  - Engineering configurations (e.g., number of antennas, transmission power).
  - Energy-saving methods like sleep modes or adaptive configurations.
- Provide actionable insights to inform more efficient RAN operations.

## Dataset

- Collected/curated data from simulated or real-world 5G network environments.
- Features include:
  - Traffic volume
  - Number of connected users
  - Power configuration settings
  - Time of day / load profiles
  - Energy-saving mode status
- Target variable: Energy consumption (in kWh or equivalent metric)

## Tools & Technologies

- **Python**
- **pandas**, **NumPy** – Data manipulation
- **scikit-learn**, **XGBoost**, **LightGBM** – Machine learning models
- **Matplotlib**, **Seaborn** – Visualization
- **Jupyter Notebook** – Interactive development

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/TechLeoo/Predictive-Modeling-for-5G-Energy-Consumption.git
cd Predictive-Modeling-for-5G-Energy-Consumption
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

---

## Usage

- Load and explore the dataset in the provided notebook.
- Preprocess the data (scaling, encoding, etc.).
- Train regression models (e.g., Random Forest, XGBoost).
- Evaluate performance using metrics such as RMSE, MAE, and R².
- Visualize feature importance and predictions.

---

## Skills Learned

- Time series and multivariate data modeling  
- Feature engineering for energy-related data  
- Regression modeling with real-world datasets  
- Energy efficiency analysis in telecom  
- Model performance evaluation and tuning  

---

## Contributing

Contributions are welcome!  
Feel free to **fork the repo**, make improvements, and submit a **pull request**.

---

> **Disclaimer:** The dataset and use case are for research and educational purposes only.

