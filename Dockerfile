# Use the official PyTorch image as the base image
FROM pytorch/pytorch:latest

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install required Python packages
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the FastAPI application code
COPY app.py .

# Expose the port on which the application will run
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
