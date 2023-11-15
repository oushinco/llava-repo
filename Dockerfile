# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Set the Transformers cache directory to /app/cache (or any other writable path)
ENV TRANSFORMERS_CACHE /tmp/cache

# Copy the requirements.txt file into the container
COPY requirements.txt ./

# Install project dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install poetry (Python package manager)
# RUN pip install poetry

# Copy the pyproject.toml and pyproject.lock files into the container
COPY pyproject.toml ./

# Install project dependencies using poetry
# RUN poetry install --no-root --no-dev

# Copy the rest of the application code into the container
COPY . .

# Change ownership and permissions of the /app directory
RUN chmod -R 777 /app


# Set permissions for specific directories (adjust as needed)
RUN mkdir /tmp/cache && chmod -R 777 /tmp/cache

# Copy the script to /tmp
# COPY start_service.sh /tmp/start_service.sh

# Make the start_service.sh script executable if needed
# RUN chmod +x start_service.sh

# Expose the port on which your service runs (7860)
EXPOSE 7860

# Run the start_service.sh script or your application's entry point
CMD ["./start_service.sh"]
