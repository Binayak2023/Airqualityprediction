import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np

# Load the saved models and scalers
gp_regressor = joblib.load("models/gp_regressor.pkl")
gp_classifier = joblib.load("models/gp_classifier.pkl")
scaler_X = joblib.load("models/scaler_X.pkl")
scaler_y = joblib.load("models/scaler_y.pkl")

# Create the main GUI window
root = tk.Tk()
root.title("Air Quality Prediction")
root.geometry("400x500")
root.configure(bg="#e6f7ff")

# Create input fields for pollutant values
tk.Label(root, text="Air Quality Prediction", font=("Arial", 20, "bold"), bg="#e6f7ff").pack(pady=10)

tk.Label(root, text="Enter the pollutant values:", font=("Arial", 12), bg="#e6f7ff").pack(pady=5)

fields = [
    ("SO2:", "entry_so2"),
    ("NO2:", "entry_no2"),
    ("CO:", "entry_co"),
    ("O3:", "entry_o3"),
    ("PM10:", "entry_pm10"),
    ("TEMP:", "entry_temp"),
    ("WSPM:", "entry_wspm"),
]

entries = {}

for field, var_name in fields:
    frame = tk.Frame(root, bg="#e6f7ff")
    frame.pack(pady=3)
    label = tk.Label(frame, text=field, font=("Arial", 12), width=10, anchor="w", bg="#e6f7ff")
    label.pack(side="left")
    entry = tk.Entry(frame, font=("Arial", 12), width=20)
    entry.pack(side="left")
    entries[var_name] = entry

# Function to get inputs and make predictions
def predict():
    try:
        # Get values from the entry fields and convert to float
        so2 = float(entries["entry_so2"].get().strip())
        no2 = float(entries["entry_no2"].get().strip())
        co = float(entries["entry_co"].get().strip())
        o3 = float(entries["entry_o3"].get().strip())
        pm10 = float(entries["entry_pm10"].get().strip())
        temp = float(entries["entry_temp"].get().strip())
        wspm = float(entries["entry_wspm"].get().strip())

        # Create a feature array
        input_features = np.array([[so2, no2, co, o3, pm10, temp, wspm]])

        # Scale the input features
        input_features_scaled = scaler_X.transform(input_features)

        # Predict PM2.5
        y_pred_reg_scaled = gp_regressor.predict(input_features_scaled)
        y_pred_reg = scaler_y.inverse_transform(y_pred_reg_scaled.reshape(-1, 1))[0][0]

        # Predict air quality classification
        y_pred_class = gp_classifier.predict(input_features_scaled)[0]

        # Display the predictions
        result_pm25.config(text=f"{y_pred_reg:.2f}")
        result_class.config(text=f"{y_pred_class}")

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values for all fields.")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")

# Create the Predict button
predict_button = tk.Button(root, text="Predict", font=("Arial", 14, "bold"), bg="#4CAF50", fg="white", command=predict)
predict_button.pack(pady=20)

# Display prediction results
tk.Label(root, text="Predicted PM2.5:", font=("Arial", 12), bg="#e6f7ff").pack(pady=5)
result_pm25 = tk.Label(root, text="", font=("Arial", 14, "bold"), bg="#e6f7ff")
result_pm25.pack(pady=5)

tk.Label(root, text="Air Quality:", font=("Arial", 12), bg="#e6f7ff").pack(pady=5)
result_class = tk.Label(root, text="", font=("Arial", 14, "bold"), bg="#e6f7ff")
result_class.pack(pady=5)

# Run the GUI loop
root.mainloop()
