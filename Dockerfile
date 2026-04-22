# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Train model inside container
RUN python src/train.py

# Expose port
EXPOSE 5000

# Run API
CMD ["python", "src/app.py"]