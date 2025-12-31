# Sentiment Analysis on IMDb Dataset

This project implements a Deep Learning model to perform sentiment analysis on movie reviews. It classifies reviews as **Positive** or **Negative** (or **Neutral** based on confidence) using a Bidirectional LSTM neural network. The project includes a Flask web application for real-time predictions.

## 📂 Project Structure

- **`main.py`**: The main script for training the model. It handles data loading, preprocessing, model training, and saving the trained model and tokenizer.
- **`app.py`**: A Flask web application that serves a simple interface for users to enter text and get sentiment predictions.
- **`src/`**: Contains the core logic:
  - `model_creation.py`: Defines the Neural Network architecture (Embedding -> Bi-LSTM -> Dense).
  - `data_preparation.py`: Handles data padding and preparation.
  - `utilis.py`: Utility functions for loading data, cleaning text, and making predictions.
  - `logger.py` & `exception.py`: Logging and custom exception handling.
- **`aclImdb/`**: Directory for the IMDb dataset (Train/Test data). *Note: This folder is expected to contain the dataset.*
- **`artifacts/`**: Stores the trained model (`imdb_review_model.keras`) and tokenizer (`tokenizer.pkl`).

## 🛠️ Technologies Used

- **Python 3.x**
- **TensorFlow / Keras**: For building and training the Deep Learning model.
- **Flask**: For the web interface.
- **Pandas & NumPy**: For data manipulation.
- **Scikit-learn**: For various utility metrics (if used).

## 🚀 Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd Sentiment_Analysis
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Prepare the Dataset:**
    - Download the **Large Movie Review Dataset (v1.0)** from [here](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz).
    - Extract the contents into the root directory so that you have an `aclImdb` folder containing `train` and `test` subdirectories.

## 🧠 Training the Model

To train the model from scratch:

```bash
python main.py
```

This script will:
1. Load reviews from the `aclImdb` folder.
2. Clean and preprocess the text.
3. Tokenize and pad the sequences.
4. Train the Bidirectional LSTM model with Early Stopping.
5. Save the trained model to `artifacts/imdb_review_model.keras`.

## 🌐 Running the Web App

Once the model is trained (or if you already have the artifacts), you can run the Flask application:

```bash
python app.py
```

- Open your browser and go to `http://127.0.0.1:5000/`.
- Enter a movie review and click "Predict" to see the sentiment.

## 📊 Model Architecture

The model consists of the following layers:
1.  **Embedding Layer**: Converts words into dense vectors of size 128 (Vocab size: 20,000).
2.  **Bidirectional LSTM**: 128 units with dropout (0.3) to capture context from both directions.
3.  **Dense Layer**: 1 unit with Sigmoid activation for binary classification.

Optimizer: **Adam** (learning rate: 1e-4)  
Loss Function: **Binary Crossentropy**


