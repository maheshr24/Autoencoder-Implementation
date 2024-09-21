# Use Python base image
FROM python:3

# Set the working directory inside the container
WORKDIR /app

# Copy the contents of the current directory into the container
COPY . .

# Install necessary Python packages
RUN pip install numpy
RUN pip install pandas
RUN pip install matplotlib
RUN pip install pillow
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


# Run your app (assuming you have a Python app to run)
CMD ["python3", "autoencoders.py"]  

