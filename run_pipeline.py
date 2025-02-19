import os
import subprocess

def run_command(command):
    result = subprocess.run(command, shell=True, check=True)
    if result.returncode != 0:
        print(f"Error: {command} failed")
        exit(1)

# Set environment variables (passed from Jenkins)
PYTHON = os.getenv("PYTHON", "python3")
MODEL = os.getenv("MODEL", "RF")

# Ensure required directories exist
os.makedirs("models", exist_ok=True)

# Train the model
run_command(f"{PYTHON} main.py train --train_data churn-bigml-80.csv --model {MODEL} --save_model models/{MODEL}.pkl")

# Test the model
run_command(f"{PYTHON} main.py test --test_data churn-bigml-20.csv --load_model models/{MODEL}.pkl")

# Run predictions (uncomment when the predict function is ready)
# run_command(f"{PYTHON} main.py predict --load_model models/{MODEL}.pkl --test_data churn-bigml-20.csv")

print("✅ Pipeline executed successfully!")
