# Use a slim Python image as the base
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt to the working directory
COPY requirements.txt /app/

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . /app/

# Expose the MLflow UI port
EXPOSE 5000

# Run the training script followed by starting the MLflow UI
CMD ["sh", "-c", "python train.py && mlflow ui --host 0.0.0.0 --port 5000"]
