import csv
from hmm_core import parse_conllu
import time

def normalize(counts, smoothing=1, num_states=0):
    """
    Function to normalize raw counts into probabilities with optional smoothing.

    :param counts: Dictionary of raw counts
    :param smoothing: Smoothing factor to avoid zero probabilities
    :param num_states: Total number of possible states (used for smoothing)
    :return: Dictionary of normalized probabilities
    """
    total = sum(counts.values()) + smoothing * num_states  # Calculate the normalization denominator
    probabilities = {key: (value + smoothing) / total for key, value in counts.items()}  # Apply smoothing
    return probabilities


def estimate_hmm(sentences):
    """
    Function to estimate HMM parameters from training data.

    :param sentences: List of (words, tags) tuples where:
                      - words: List of words in a sentence
                      - tags: Corresponding list of POS tags for the words
    :return: Tuple containing:
             - Starting probabilities: P(tag | start)
             - Transition probabilities: P(next_tag | current_tag)
             - Emission probabilities: P(word | tag)
             - Total counts for each tag
    """
    # Initialize count dictionaries
    start_counts = {}  # Count occurrences of tags at the beginning of sentences
    trans_counts = {}  # Count transitions between tags
    emit_counts = {}   # Count emissions (word given tag)
    tag_totals = {}    # Track the total occurrences of each tag

    # Iterate over the sentences in the training data
    for words, tags in sentences:
        for i, (word, tag) in enumerate(zip(words, tags)):
            # Update the total counts for each tag
            tag_totals[tag] = tag_totals.get(tag, 0) + 1

            # Update emission counts: P(word | tag)
            if tag not in emit_counts:
                emit_counts[tag] = {}
            emit_counts[tag][word] = emit_counts[tag].get(word, 0) + 1

            # Handle starting tags: P(tag | start)
            if i == 0:
                start_counts[tag] = start_counts.get(tag, 0) + 1
            else:
                # Update transition counts: P(next_tag | current_tag)
                prev_tag = tags[i - 1]
                if prev_tag not in trans_counts:
                    trans_counts[prev_tag] = {}
                trans_counts[prev_tag][tag] = trans_counts[prev_tag].get(tag, 0) + 1

    # Normalize counts to probabilities
    start_probs = normalize(start_counts)  # Starting probabilities
    trans_probs = {tag: normalize(trans, smoothing=1, num_states=len(tag_totals))  # Transition probabilities
                   for tag, trans in trans_counts.items()}
    emit_probs = {tag: normalize(emits)  # Emission probabilities
                  for tag, emits in emit_counts.items()}

    return start_probs, trans_probs, emit_probs, tag_totals  # Return all parameters


def save_csv(file_name, headers, rows):
    """
    Function to save data to a CSV file.

    :param file_name: Name of the output CSV file
    :param headers: List of column headers
    :param rows: List of rows, where each row is a list of values
    """
    with open(file_name, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)  # Write the headers
        writer.writerows(rows)    # Write the data rows


def save_params(start_probs, trans_probs, emit_probs):
    """
    Function to save HMM parameters to CSV files.

    :param start_probs: Starting probabilities
    :param trans_probs: Transition probabilities
    :param emit_probs: Emission probabilities
    """
    # Save starting probabilities
    with open("start_probs.csv", "w", encoding="utf-8") as f:
        for tag, prob in start_probs.items():
            f.write(f"{tag},{prob}\n")

    # Save transition probabilities
    with open("trans_probs.csv", "w", encoding="utf-8") as f:
        for from_tag, to_tags in trans_probs.items():
            for to_tag, prob in to_tags.items():
                f.write(f"{from_tag},{to_tag},{prob}\n")

    # Save emission probabilities
    with open("emit_probs.csv", "w", encoding="utf-8") as f:
        for tag, words in emit_probs.items():
            for word, prob in words.items():
                f.write(f"{tag},{word},{prob}\n")


def train_hmm(train_file):
    """
    Function to train the HMM model by estimating parameters from a training file.

    :param train_file: Path to the training file in CoNLL-U format
    :return: Tuple containing HMM parameters (start, transition, and emission probabilities)
    """
    print(f"Parsing training data from {train_file}...")  # Notify the user about data parsing
    sentences = parse_conllu(train_file)  # Parse the CoNLL-U file to extract words and tags
    return estimate_hmm(sentences)  # Estimate HMM parameters


def main():
    """
    Main function to train the HMM and save the parameters to CSV files.
    """
    train_file = "de_gsd-ud-train.conllu"  # Path to the training file
    print("Training HMM...")  # Notify the user about training

    # Measure training time
    start_time = time.time()
    start_probs, trans_probs, emit_probs, tag_totals = train_hmm(train_file)  # Train the HMM
    end_training = time.time()

    print("Saving parameters...")  # Notify the user about saving
    save_params(start_probs, trans_probs, emit_probs)  # Save parameters to CSV files
    end_saving = time.time()

    # Notify the user about completion
    print("Training and saving completed.")
    print(f"  Start probabilities: saved to start_probs.csv")
    print(f"  Transition probabilities: saved to trans_probs.csv")
    print(f"  Emission probabilities: saved to emit_probs.csv")
    print("\nExecution Times:")
    print(f"  Training time: {end_training - start_time:.6f} seconds")
    print(f"  Saving time: {end_saving - end_training:.6f} seconds")
    print(f"  Total time: {end_saving - start_time:.6f} seconds")


if __name__ == "__main__":
    main()

