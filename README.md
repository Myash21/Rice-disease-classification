# Rice Disease Classification and Recommendation System

This project uses a deep learning model to classify four types of rice diseases—Bacterial Blight, Blast Disease, Brown Spot, and False Smut. Leveraging PyTorch and Flask, this web-based tool allows users to upload rice crop images for disease diagnosis, aiming to aid farmers and agronomists in timely disease identification and management.

## Features

- **Disease Classification**: Identifies four common rice diseases from image inputs.
- **Web-based Interface**: Built with Flask, allowing easy access to predictions via a web interface.
- **Custom Model Training**: Utilizes a fine-tuned ResNet-18 model.
- **Real-time Image Processing**: Processes images in real-time to deliver fast predictions.

## Dataset

The model was trained on a curated rice disease dataset with images labeled for each disease type. The dataset is split into training and validation subsets with standard transformations for data augmentation.

## Model

The classification model is based on the ResNet-18 architecture with modifications to the fully connected layer to accommodate four disease classes.

- **Architecture**: ResNet-18
- **Classes**: Bacterial Blight, Blast, Brown Spot, and False Smut
- **Preprocessing**: Resizing, cropping, and normalization
- **Training Framework**: PyTorch

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/rice-disease-classification.git
   cd rice-disease-classification
2. **Install dependencies**:
      ```bash
   pip install -r requirements.txt

## Usage

1. **Run the Flask app**:
      ```bash
   python app.py
2. Access the web interface: Open http://localhost:5000 in your browser.
3. Upload and Predict:
   - Upload an image of the rice plant.
   - The app will classify the image and display the predicted disease name.
  
## Code Overview
- app.py: Defines the Flask app, handling web routes and model predictions.
- model.py: Contains code for training the model, including data preprocessing and model tuning.
- requirements.txt: Lists the required Python packages.
- templates/index.html: Frontend template for uploading images and viewing predictions.

## Results
The model achieves high accuracy on validation data, with detailed performance metrics reported on the training dataset. Example results are provided for each disease category.

## Future Work
- Expanding to include additional rice diseases.
- Integrating recommendation logic for disease management.
- Enhancing the user interface for mobile accessibility.
