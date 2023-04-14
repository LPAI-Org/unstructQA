# Use the official Python 3.10.10 image as the base image
FROM python:3.10.10

# Set the working directory
WORKDIR /app

# Copy the requirements files into the container
COPY base_requirements.txt detectron_requirements.txt ./

# Create a virtual environment called 'en'
RUN python -m venv en

# Activate the virtual environment and install the required packages
RUN /bin/bash -c "source en/bin/activate && \
    pip install --no-cache-dir -r base_requirements.txt && \
    pip install --no-cache-dir -r detectron_requirements.txt && \
    pip uninstall -y protobuf && \
    pip install protobuf==3.19.3"

# Copy the app.py & env api keys file into the container
COPY app.py ./
COPY .env ./

# Set the default command to run when the container starts
CMD ["/app/en/bin/python", "/app/app.py"]