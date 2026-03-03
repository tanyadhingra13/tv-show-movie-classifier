# Media Content Genre Classifier

This project implements a multi-label classification pipeline to predict genres (e.g., "Documentaries", "TV Sci-Fi & Fantasy") for media content based on features like release year, content type, rating, and platform.

The project compares two approaches: **Logistic Regression** and **Random Forest**, using a One-Vs-Rest strategy to handle multiple genre labels per entry.

## 📊 Results Summary

Based on the latest evaluation:
| Model | F1 Micro | F1 Macro | Hamming Loss |
| :--- | :--- | :--- | :--- |
| **Random Forest** | 0.5008 | 0.2348 | 0.0226 |
| **Logistic Regression** | 0.4834 | 0.1624 | 0.0206 |

### Feature Importance
The Random Forest model relies heavily on `release_year`, followed by the content `type` (Movie vs. TV Show) and the streaming `platform`.


---

## 📂 Project Structure

* `data_loader.py`: Utility to load CSV data.
* `preprocess.py`: Handles multi-label binarization, feature scaling, and one-hot encoding.
* `model.py`: Defines the model architectures (Logistic Regression and Random Forest).
* `train.py`: Orchestrates the training pipeline and saves the model.
* `evaluate.py`: Calculates performance metrics (F1-score, Hamming Loss).
* `explain.py`: Generates feature importance visualizations.
* `outputs/`: Directory containing saved models (`.pkl`), metrics (`.json`), and plots (`.png`).

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt


### 2. Training a Model
To train the Random Forest model: python train.py --data_path your_data.csv --model_name random_forest

### 3. Evaluation
To evaluate the trained model and save metrics: python evaluate.py --model_path outputs/model_random_forest.pkl --data_path your_data.csv --model_name random_forest

### 4. Explainability
To generate the feature importance plot: python explain.py --model_path outputs/model_random_forest.pkl --model_name random_forest
















