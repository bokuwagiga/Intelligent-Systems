import random


def run_classifier(print_output=False):
    w1 = random.random()
    w2 = random.random()
    b = random.random()
    eta = random.random()

    training_items = []
    testing_items = []

    with open('Data.txt', 'r') as f:
        lines = f.read().splitlines()
        for ind in range(len(lines)):
            item = tuple([float(item) for item in lines[ind].split(',')])
            if ind % 2 == 0:
                training_items.append(item)
            else:
                testing_items.append(item)

    abs_error = 1
    iter_num = 0
    output_lines = []

    output_lines.append("\nStarting Training...\n")

    while abs_error != 0:
        iter_num += 1
        abs_error = 0
        output_lines.append(f"--- Iteration {iter_num} ---\n")

        for item in training_items:
            x1, x2, d = item
            ws = x1 * w1 + x2 * w2 + b
            y = 1 if ws > 0 else -1
            error = d - y
            abs_error += abs(error)

            output_lines.append(f"Input: x1 = {x1}, x2 = {x2}\n")
            output_lines.append(f"Weighted Sum (ws): {ws:.4f}\n")
            output_lines.append(f"Expected: {'Apple' if d == 1 else 'Pear'}\n")
            output_lines.append(f"Predicted: {'Apple' if y == 1 else 'Pear'}\n")
            output_lines.append(f"Error: {error}\n")
            output_lines.append(f"Prediction Correct: {'Yes' if d == y else 'No'}\n")
            output_lines.append("-" * 30 + "\n")

            w1 += eta * error * x1
            w2 += eta * error * x2
            b += eta * error

    output_lines.append("\nTraining Complete!\n")
    output_lines.append(f"Final Weights and Bias after {iter_num} iterations:\n")
    output_lines.append(f"w1: {w1:.4f}\n")
    output_lines.append(f"w2: {w2:.4f}\n")
    output_lines.append(f"b: {b:.4f}\n")

    correct_predictions = 0
    output_lines.append("\nStarting Testing...\n")

    for item in testing_items:
        x1, x2, d = item
        ws = x1 * w1 + x2 * w2 + b
        y = 1 if ws > 0 else -1

        output_lines.append(f"Input: x1 = {x1}, x2 = {x2}\n")
        output_lines.append(f"Expected: {'Apple' if d == 1 else 'Pear'}\n")
        output_lines.append(f"Predicted: {'Apple' if y == 1 else 'Pear'}\n")
        output_lines.append(f"Prediction Correct: {'Yes' if d == y else 'No'}\n")
        output_lines.append("-" * 30 + "\n")

        if d == y:
            correct_predictions += 1

    accuracy = (correct_predictions / len(testing_items) * 100) if testing_items else 0
    output_lines.append(f"\nTesting Complete! Accuracy: {accuracy:.2f}%\n")

    output_lines.append("\n--- Analysis and Conclusion ---\n")
    output_lines.append(f"The algorithm completed training in {iter_num} iterations.\n")
    if accuracy == 100:
        output_lines.append(
            "The model perfectly classified all testing items, indicating that it has learned the patterns well.\n")
    elif accuracy >= 80:
        output_lines.append("The model performed well with high accuracy, indicating that it generalizes well.\n")
    else:
        output_lines.append("The model struggled with testing, indicating it needs better tuning or more training.\n")
    output_lines.append("--- End of analysis. ---\n")

    with open('results.txt', 'w') as results:
        for output_line in output_lines:
            results.write(output_line)
            if print_output:
                print(output_line, end='')

    return accuracy, iter_num


eventual_accuracy = 0
execution_results = []

number_of_execution = 150
for i in range(number_of_execution):
    exec_num = i + 1
    print_output = (i == number_of_execution - 1)
    accuracy, iterations = run_classifier(print_output=print_output)
    execution_results.append((iterations, accuracy))

RED = "\033[91m"
RESET = "\033[0m"

print(f'\n{RED}Please note that the output above is only for the last execution of the model!\n'
      f'All results are saved in the results.txt file.{RESET}\n')

if execution_results:
    avg_iterations = sum(iterations for iterations, _ in execution_results) / len(execution_results)
    avg_accuracy = sum(accuracy for _, accuracy in execution_results) / len(execution_results)

    print(f'--- After {number_of_execution} executions of the model ---\n'
          f'Average number of training iterations: {avg_iterations:.2f}\n'
          f'Average testing accuracy: {avg_accuracy:.2f}%')

else:
    print("No executions were completed.")
