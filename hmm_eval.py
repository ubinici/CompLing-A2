import time
from hmm_core import viterbi, parse_conllu
from hmm_train import train_hmm
from collections import defaultdict


def predict(test_data, start_probs, trans_probs, emit_probs, tag_totals):
    """
    Function to predict POS tags using the Viterbi algorithm.

    :param test_data: List of (words, gold_tags) tuples from the test set
    :param start_probs: Starting probabilities for each tag
    :param trans_probs: Transition probabilities between tags
    :param emit_probs: Emission probabilities for words given tags
    :param tag_totals: Total counts for each tag (used for smoothing)
    :return: List of predicted tag sequences for each test sentence
    """
    results = []
    for words, _ in test_data:
        # Use the Viterbi algorithm to compute the best sequence of tags
        tags, _ = viterbi(words, start_probs, trans_probs, emit_probs, tag_totals)
        results.append(tags)
    return results


def evaluate(predictions, test_data):
    """
    Function to evaluate the accuracy of predictions against the gold standard.

    :param predictions: List of predicted tag sequences
    :param test_data: List of (words, gold_tags) tuples from the test set
    :return: Tuple containing accuracy and a dictionary of error counts
    """
    total, correct = 0, 0
    errors = defaultdict(int)  # Track mispredictions as (gold_tag, pred_tag)

    for preds, (_, golds) in zip(predictions, test_data):
        for pred, gold in zip(preds, golds):
            total += 1
            if pred == gold:
                correct += 1
            else:
                errors[(gold, pred)] += 1  # Increment error count for the pair

    accuracy = correct / total if total > 0 else 0.0  # Avoid division by zero
    return accuracy, errors


def diagnostics(errors, top_n=10):
    """
    Function to display the most common prediction errors.

    :param errors: Dictionary mapping (gold_tag, pred_tag) pairs to error counts
    :param top_n: Number of top errors to display
    """
    print("\nTop Prediction Errors:")
    sorted_errors = sorted(errors.items(), key=lambda x: x[1], reverse=True)  # Sort by error count
    for (gold, pred), count in sorted_errors[:top_n]:
        print(f"  Gold: {gold}, Predicted: {pred}, Count: {count}")


def main(train_path, test_path):
    """
    Main function to train the HMM, make predictions, and evaluate them.

    :param train_path: Path to the training data file in CoNLL-U format
    :param test_path: Path to the testing data file in CoNLL-U format
    """
    # Measure overall execution time
    start_time = time.time()

    # Step 1: Train the HMM
    print("Training HMM...")
    start_probs, trans_probs, emit_probs, tag_totals = train_hmm(train_path)
    train_time = time.time()

    # Step 2: Load and parse test data
    print("Loading test data...")
    test_data = parse_conllu(test_path)
    load_time = time.time()

    # Step 3: Predict POS tags using the trained HMM
    print("Predicting POS tags...")
    predictions = predict(test_data, start_probs, trans_probs, emit_probs, tag_totals)
    predict_time = time.time()

    # Step 4: Evaluate the predictions
    print("Evaluating predictions...")
    accuracy, errors = evaluate(predictions, test_data)
    eval_time = time.time()

    # Display the results
    print(f"\nAccuracy: {accuracy:.4f}")
    diagnostics(errors)

    # Display timing information
    print("\nExecution Times:")
    print(f"  Training: {train_time - start_time:.6f} seconds")
    print(f"  Loading Test Data: {load_time - train_time:.6f} seconds")
    print(f"  Prediction: {predict_time - load_time:.6f} seconds")
    print(f"  Evaluation: {eval_time - predict_time:.6f} seconds")
    print(f"  Total Time: {eval_time - start_time:.6f} seconds")


if __name__ == "__main__":
    # Define paths to the training and testing data files
    train_file = "de_gsd-ud-train.conllu"
    test_file = "de_gsd-ud-test.conllu"
    main(train_file, test_file)
