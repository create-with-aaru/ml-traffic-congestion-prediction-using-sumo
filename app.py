import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def generate_sumo_features(n_samples=1500):
    """Simulates edge loop detector metrics extracted via SUMO."""
    np.random.seed(42)
    vehicle_count = np.random.randint(5, 120, size=n_samples)
    density = np.clip(vehicle_count / 1.5 + np.random.normal(0, 3, size=n_samples), 1, 100)
    speed = np.maximum(5.0, 60.0 - (density * 0.55) + np.random.normal(0, 4, size=n_samples))
    waiting_time = np.maximum(0.0, (density * 1.8) - (speed * 0.8) + np.random.normal(0, 5, size=n_samples))

    # Congestion label: 0 = Low, 1 = Moderate, 2 = High
    score = (density * 0.5) + (waiting_time * 0.3) - (speed * 0.2)
    labels = np.where(score < 25, 0, np.where(score < 50, 1, 2))

    return pd.DataFrame({
        "vehicle_count": vehicle_count,
        "density_veh_km": density,
        "mean_speed_kmh": speed,
        "waiting_time_sec": waiting_time,
        "congestion_level": labels
    })

def main():
    print("1. Ingesting SUMO extracted features...")
    df = generate_sumo_features()
    
    X = df[["vehicle_count", "density_veh_km", "mean_speed_kmh", "waiting_time_sec"]]
    y = df["congestion_level"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("2. Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    print("3. Evaluating Model:")
    y_pred = model.predict(X_test_scaled)
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=["Low", "Moderate", "High"]))

if __name__ == "__main__":
    main()
