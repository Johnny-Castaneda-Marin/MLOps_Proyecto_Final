FROM locustio/locust:2.29.1

# Copy the locustfile into the container
COPY locustfile.py /mnt/locust/locustfile.py

# Set the working directory
WORKDIR /mnt/locust

# Default command: run locust in headless mode targeting the API service
# These can be overridden via environment variables or K8s args
ENTRYPOINT ["locust", "-f", "/mnt/locust/locustfile.py"]
