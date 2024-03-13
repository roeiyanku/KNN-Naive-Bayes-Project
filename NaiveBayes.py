# Made by:
# Roei Yanku - 207014440
# Please Plug in your directory in line 12!!
# This code uses the Naive Bayes algorithm to classify tweets of flight reviews as
# either positive or negative

import pandas as pd
import numpy as np
import string

# Load the data from a CSV file
df = pd.read_csv(r'Please-Provide-Directory-Here\flightsgoodbad.csv')


# Ignore blank spaces
df.dropna(inplace=True)

# Preprocess text by removing punctuation and converting to lowercase
df['text'] = df['text'].apply(lambda x: ' '.join([word.strip(string.punctuation).lower() for word in x.split()]))

# Split the data into training and test sets
train_size = int(0.8 * len(df))
train_data = df[:train_size]
test_data = df[train_size:]

# Calculate class priors
class_priors = train_data['airline_sentiment'].value_counts(normalize=True).to_dict()

# Calculate conditional probabilities
word_counts = {}
for label in class_priors:
    text = ' '.join(train_data[train_data['airline_sentiment'] == label]['text'])
    tokens = text.split()
    word_counts[label] = pd.Series(tokens).value_counts().to_dict()

# Laplace smoothing parameter
alpha = 1

# Function to classify a new text
def classify(text):
    if isinstance(text, str):  # Check if text is a string
        tokens = [word.strip(string.punctuation).lower() for word in text.split()]
        class_probs = {}
        for label in class_priors:
            prob = np.log(class_priors[label])
            for word in tokens:
                word_prob = (word_counts[label].get(word, 0) + alpha) / (len(word_counts[label]) + alpha * len(tokens))
                prob += np.log(word_prob)
            class_probs[label] = prob
        return max(class_probs, key=class_probs.get)
    else:
        return 'neutral'  # Return 'neutral' for non-string values


# Classify test data and calculate accuracy, precision, recall, and F-measure
predicted_labels = test_data['text'].apply(classify)
actual_labels = test_data['airline_sentiment']

# Classify train data and calculate accuracy
train_predicted_labels = train_data['text'].apply(classify)
train_actual_labels = train_data['airline_sentiment']
accuracy_train = sum(train_predicted_labels == train_actual_labels) / len(train_actual_labels)
print(f'Accuracy on train set: {accuracy_train}')

# Classify test data and calculate accuracy
test_predicted_labels = test_data['text'].apply(classify)
test_actual_labels = test_data['airline_sentiment']
accuracy_test = sum(test_predicted_labels == test_actual_labels) / len(test_actual_labels)
print(f'Accuracy on test set: {accuracy_test}')


accuracy = sum(predicted_labels == actual_labels) / len(actual_labels)

TP = sum((actual_labels == 'positive') & (predicted_labels == 'positive'))
TN = sum((actual_labels == 'negative') & (predicted_labels == 'negative'))
FP = sum((actual_labels == 'negative') & (predicted_labels == 'positive'))
FN = sum((actual_labels == 'positive') & (predicted_labels == 'negative'))

# Notice! see how we focused to check for TRUE NEGATIVE (TN) instead of TP  - that is because oir
# dataset was negative heavy (about 75%)
precision = TN / (TN + FN) if TP + FP != 0 else 0
recall = TN / (TN + FP) if TP + FN != 0 else 0
f_measure = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0


print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F-measure: {f_measure}')