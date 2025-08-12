import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import pandas as pd
import joblib

# Load pre-trained model and encoder
model = joblib.load("har_model.pkl")
le = joblib.load("label_encoder.pkl")

# Prediction function
def predict_activity():
    file_path = filedialog.askopenfilename(filetypes=[("Excel or CSV files", "*.xlsx *.csv")])
    if not file_path:
        return

    try:
        # Read file
        if file_path.endswith(".csv"):
            data = pd.read_csv(file_path)
        else:
            data = pd.read_excel(file_path)

        # Drop non-feature columns if present
        features = data.drop(columns=['subject', 'Activity'], errors='ignore')

        # Predict
        predictions = model.predict(features)
        activities = le.inverse_transform(predictions)

        # Display in textbox
        text_box.delete(1.0, tk.END)
        for i, activity in enumerate(activities):
            text_box.insert(tk.END, f"Sample {i+1}: {activity}\n")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to process file:\n{str(e)}")

# Setup GUI
root = tk.Tk()
root.title("HAR Activity Predictor")

upload_button = tk.Button(root, text="Upload Excel/CSV File", command=predict_activity, width=30)
upload_button.pack(pady=10)

text_box = scrolledtext.ScrolledText(root, width=60, height=20)
text_box.pack(padx=10, pady=10)

root.mainloop()
