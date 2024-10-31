Heart Disease and Flight Review Sentiment Classification
This project uses two classification algorithms:

1. k-Nearest Neighbors (k-NN): Predicts heart disease based on health metrics.
2. Naive Bayes: Classifies flight reviews as positive or negative.

Getting Started
Requirements
  Python 3.x
  Pandas and NumPy libraries
Data
1. Heart Disease Data (heart_statlog_cleveland_hungary_final.csv)
2. Flight Review Data (flightsgoodbad.csv)
Update File Paths: Enter the correct file paths in the code at line 12 for the heart disease dataset and the corresponding line for the flight sentiment dataset.

Steps
1. Clone the Project:
   git clone https://github.com/yourusername/yourrepository.git
    cd yourrepository
2. Install Dependencies:
   pip install pandas numpy
3. Run the Script
   python script_name.py
4. Results: The script outputs accuracy, precision, recall, and F-measure for both training and test sets.

   Project Details
k-Nearest Neighbors: Predicts heart disease by measuring the Euclidean distance between data points.
Naive Bayes: Classifies flight sentiment using text preprocessing and Laplace smoothing.
