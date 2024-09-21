# Autoencoder



This project implements an autoencoder in PyTorch to process and reconstruct images. The repository contains the necessary scripts and files for training and running the autoencoder, along with a Docker setup for running the entire project inside a container.

## Files and Directories

- `.dockerignore`: Specifies files and directories to be ignored when building the Docker image.
- `Dockerfile`: Contains instructions to build the Docker image for running the autoencoder model.
- `README.md`: Documentation file providing an overview of the project.
- `autoencoders.py`: Main Python script that defines the autoencoder model and loads an image for reconstruction.
- `image.jpg`: Input image used for testing the autoencoder's performance.
- `model.pth`: Pre-trained model file containing the weights of the autoencoder.
- `output.png`: Reconstructed image produced by the autoencoder after processing `image.jpg`.

## Project Overview

The main goal of this project is to demonstrate how autoencoders can be used for image reconstruction. The autoencoder model is trained on images, and given an input image, it attempts to reconstruct it by compressing and decompressing the input.

### Autoencoder Architecture

- The autoencoder consists of an encoder and a decoder.
- The encoder reduces the dimensionality of the input image.
- The decoder reconstructs the image from the compressed representation.
- The model uses convolutional layers in both the encoder and decoder stages, making it suitable for image data.

## Requirements

- Python 3.x
- Docker
- PyTorch and torchvision
- Pillow (PIL)
- Matplotlib
- Numpy

## Usage

### 1. Running the Project Locally

To run the project locally, you need to install the necessary dependencies and run the Python script:

1. Clone the repository:

   ```bash
   git clone https://github.com/mahesh24/autoencoder-image.git
   cd autoencoder-image
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Python script:

   ```bash
   python autoencoders.py
   ```

4. The reconstructed image will be saved as `output.png` in the root directory.

### 2. Running the Project with Docker

You can also run the project using Docker by building and running the Docker container:

1. Build the Docker image:

   ```bash
   docker build -t autoencoder-image .
   ```

2. Run the container:

   ```bash
   docker run autoencoder-image
   ```

3. The reconstructed image will be saved as `output.png` inside the container. You can either use volume mounting or `docker cp` to access the file from your host machine.

### 3. Accessing the Output Image

If you used Docker, you can copy the output image from the container to your local machine:

```bash
docker cp <container_id>:/app/output.png /path/to/local/output.png
```

## Notes

- The pre-trained model (`model.pth`) is loaded in the script, and the input image (`image.jpg`) is processed for reconstruction.
- Ensure that `model.pth` and `image.jpg` are placed in the correct directories as mentioned.

## License

This project is open source and available under the [MIT License](LICENSE).

---
