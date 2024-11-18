import time
import matplotlib.pyplot as plt
from hmm_core import parse_conllu
from hmm_train import estimate_hmm
from hmm_eval import predict, evaluate


def learning_curve(train_file, test_file, step_size):
    """
    Function to compute the learning curve by training on progressively larger subsets of data.

    :param train_file: Path to the training dataset (CoNLL-U format)
    :param test_file: Path to the test dataset (CoNLL-U format)
    :param step_size: Step size for the number of sentences to add in each training subset
    """
    print("Loading datasets...")  # Notify the user about dataset loading
    full_train = parse_conllu(train_file)  # Parse the full training dataset
    test_data = parse_conllu(test_file)    # Parse the test dataset

    sizes, accuracies = [], []  # Initialize lists for storing training sizes and corresponding accuracies
    total_train_time, total_eval_time = 0, 0  # Track cumulative training and evaluation times

    print("\nGenerating the learning curve...")
    # Iterate through subsets of the training data
    for size in range(step_size, len(full_train) + 1, step_size):
        print(f"\nTraining with {size} sentences...")

        # Train HMM on the current subset
        train_start = time.time()
        init_probs, trans_probs, emit_probs, tag_counts = estimate_hmm(full_train[:size])  # Train on a subset
        train_end = time.time()
        total_train_time += train_end - train_start

        # Predict and evaluate using the test data
        eval_start = time.time()
        preds = predict(test_data, init_probs, trans_probs, emit_probs, tag_counts)  # Generate predictions
        acc, _ = evaluate(preds, test_data)  # Calculate accuracy
        eval_end = time.time()
        total_eval_time += eval_end - eval_start

        print(f"Accuracy: {acc:.4f}")  # Display accuracy for the current subset

        # Record training size and accuracy for plotting
        sizes.append(size)
        accuracies.append(acc)

    # Display total training and evaluation times
    print(f"\nTotal Training Time: {total_train_time:.6f} seconds")
    print(f"Total Evaluation Time: {total_eval_time:.6f} seconds")

    # Plot the learning curve
    plot_curve(sizes, accuracies)


def plot_curve(sizes, accuracies):
    """
    Function to plot the learning curve.

    :param sizes: List of training data sizes
    :param accuracies: List of accuracies corresponding to the training sizes
    """
    print("\nPlotting the learning curve...")  # Notify the user about plotting
    plt.figure(figsize=(10, 5))  # Create a figure for the plot
    plt.plot(sizes, accuracies, marker="o", label="Accuracy")  # Plot accuracy vs. training size
    plt.title("Learning Curve: Training Size vs. Accuracy")  # Add a title
    plt.xlabel("Training Size")  # Label the x-axis
    plt.ylabel("Accuracy")       # Label the y-axis
    plt.legend()                 # Add a legend
    plt.grid(True)               # Add a grid for readability
    plt.show()                   # Display the plot


if __name__ == "__main__":
    # Paths to the training and testing datasets
    train_file = "de_gsd-ud-train.conllu"
    test_file = "de_gsd-ud-test.conllu"
    step_size = 500  # Define the step size for training subsets

    print("Starting Learning Curve Computation...\n")  # Notify the user about the process start
    start_time = time.time()  # Record the overall start time
    learning_curve(train_file, test_file, step_size)  # Compute the learning curve
    end_time = time.time()  # Record the overall end time
    print(f"\nTotal Runtime: {end_time - start_time:.6f} seconds")  # Display total runtime
