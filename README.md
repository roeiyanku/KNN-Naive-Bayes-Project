# Heart Disease and Flight Review Sentiment Classification

This project demonstrates two classification algorithms applied to distinct tasks:

1. **k-Nearest Neighbors (k-NN)**: Predicts the likelihood of heart disease based on various health metrics.
2. **Naive Bayes**: Classifies flight reviews as either positive or negative sentiment.

## Getting Started

### Requirements
- **Python 3.x**
- **Libraries**: Pandas, NumPy

### Datasets
1. **Heart Disease Data**: `heart_statlog_cleveland_hungary_final.csv`
2. **Flight Review Data**: `flightsgoodbad.csv`

> **File Paths**: Update the dataset paths in the code at line 12 for the heart disease dataset and the corresponding line for the flight sentiment dataset.

### Setup and Execution

1. **Clone the Project**:
   ```bash
   git clone https://github.com/yourusername/yourrepository.git
   cd yourrepository
   ```

2. **Install Dependencies**:
   ```bash
   pip install pandas numpy
   ```

3. **Run the Script**:
   ```bash
   python script_name.py
   ```

4. **Results**:
   - The script will output accuracy, precision, recall, and F-measure for both training and test sets, providing insight into the model’s performance.

## Project Details

### 1. k-Nearest Neighbors (k-NN) for Heart Disease Prediction
   - Uses Euclidean distance to analyze health metrics and classify patients as having heart disease or not.
   - Evaluates model performance using key metrics such as accuracy, precision, recall, and F-measure.

### 2. Naive Bayes for Flight Review Sentiment Analysis
   - Applies text preprocessing techniques and Laplace smoothing to classify reviews as positive or negative.
   - Evaluates classification results based on sentiment and provides performance metrics.

