import os
import subprocess

def run_command(command):
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Error: {command} failed")
        exit(1)

# Set environment variables (these can be passed from Jenkins)
PYTHON = os.getenv("PYTHON", "python3")
MODEL = os.getenv("MODEL", "RF")

# Train the model
run_command(f"{PYTHON} main.py train --train_data churn-bigml-80.csv --model {MODEL} --save_model models/{MODEL}.pkl")

# Test the model
run_command(f"{PYTHON} main.py test --test_data churn-bigml-20.csv --load_model models/{MODEL}.pkl")

# Run predictions
#run_command(f"{PYTHON} main.py --train_data churn-bigml-80.csv --test_data churn-bigml-20.csv --model {MODEL}")

print("Pipeline executed successfully!")
