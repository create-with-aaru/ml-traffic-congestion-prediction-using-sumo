# ML-Based Traffic Congestion Prediction using SUMO

A machine learning classification pipeline designed to assess and predict urban traffic congestion using simulation metrics generated via Eclipse SUMO (Simulation of Urban MObility).

## Overview
This project simulates real-world road network dynamics in SUMO, extracts edge flow metrics (vehicle count, density, average speed, waiting time), and applies a Random Forest classifier to categorize traffic congestion levels into three distinct tiers: **Low**, **Moderate**, and **High**.

## Methodology
- **Simulation Environment**: Eclipse SUMO
- **Extracted Features**:
  - `vehicle_count`: Edge induction loop detector vehicle frequency
  - `density_veh_km`: Vehicle concentration per kilometer
  - `mean_speed_kmh`: Harmonic mean speed of traversing vehicles
  - `waiting_time_sec`: Aggregate stoppage duration at signals
- **Algorithm**: Random Forest Classifier (`scikit-learn`)
- **Metrics**: Precision, Recall, F1-Score, Overall Accuracy (~95%)

## Execution
```bash
python app.py
