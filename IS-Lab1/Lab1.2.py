import numpy as np


def run_naive_bayes():
    training_items = []
    testing_items = []

    with open('Data.txt', 'r') as f:
        lines = f.read().splitlines()
        for ind in range(len(lines)):
            item = tuple([float(value) for value in lines[ind].split(',')])
            if ind % 2 == 0:
                training_items.append(item)
            else:
                testing_items.append(item)

    training_items = np.array(training_items)
    testing_items = np.array(testing_items)

    total_count = len(training_items)
    class_labels = np.unique(training_items[:, -1])
    priors = {label: np.sum(training_items[:, -1] == label) / total_count for label in class_labels}

    likelihoods = {}
    for label in class_labels:
        class_items = training_items[training_items[:, -1] == label]
        mean = np.mean(class_items[:, :-1], axis=0)
        var = np.var(class_items[:, :-1], axis=0)
        likelihoods[label] = (mean, var)

    print("\nStarting Testing...\n")

    correct_predictions = 0

    for item in testing_items:
        x1, x2, d = item
        class_probabilities = {}

        for label in class_labels:
            mean, var = likelihoods[label]
            likelihood = np.prod(gaussian_probability(item[:-1], mean, var))
            class_probabilities[label] = priors[label] * likelihood

        predicted_class = max(class_probabilities, key=class_probabilities.get)
        print(f"Input: x1 = {x1}, x2 = {x2}")
        print(f"Expected: {'Apple' if d == 1 else 'Pear'}")
        print(f"Predicted: {'Apple' if predicted_class == 1 else 'Pear'}")
        print(f"Prediction Correct: {'Yes' if d == predicted_class else 'No'}")
        print("-" * 30 + "")

        if d == predicted_class:
            correct_predictions += 1

    accuracy = (correct_predictions / len(testing_items) * 100) if testing_items.size > 0 else 0
    print(f"\nTesting Complete! Accuracy: {accuracy:.2f}%\n")


    return accuracy


def gaussian_probability(x, mean, var):
    coefficient = 1 / np.sqrt(2 * np.pi * var)
    exponent = np.exp(-((x - mean) ** 2) / (2 * var))
    return coefficient * exponent


eventual_accuracy = 0
execution_results = []


accuracy = run_naive_bayes()



