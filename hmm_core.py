import time
from conllu import parse

def viterbi(obs, start_probs, trans_probs, emit_probs, tag_totals):
    """
    Function to compute the most likely sequence of states using the Viterbi algorithm.

    :param obs: List of observations
    :param start_probs: Dictionary of starting probabilities for each state
    :param trans_probs: Dictionary of transition probabilities between states
    :param emit_probs: Dictionary of emission probabilities for observations given states
    :param tag_totals: Total occurrences of each state
    :return: Tuple containing the most likely sequence of states and its probability
    """
    tags = list(start_probs.keys())  # List of possible states
    V = [{}]  # Probability table for dynamic programming
    path = {tag: [] for tag in tags}  # Paths to store the best sequence for each state

    # Initialization step for the first observation
    for tag in tags:
        V[0][tag] = start_probs.get(tag, 1 / len(tags)) * emit(tag, obs[0], emit_probs, tag_totals, len(tags))
        path[tag] = [tag]

    # Recursion step for subsequent observations
    for t in range(1, len(obs)):
        V.append({})  # Add a new row for the current time step
        new_path = {}

        for cur_tag in tags:  # Iterate through all possible current states
            # Compute the maximum probability and the previous state that achieves it
            prob, prev_tag = max(
                (V[t - 1][prev] * trans_probs[prev].get(cur_tag, 1e-10) *
                 emit(cur_tag, obs[t], emit_probs, tag_totals, len(tags)), prev)
                for prev in tags
            )
            V[t][cur_tag] = prob  # Store the probability
            new_path[cur_tag] = path[prev_tag] + [cur_tag]  # Update the path

        path = new_path  # Update paths to reflect the best sequences so far

    # Termination step to determine the best final state
    prob, state = max((V[-1][tag], tag) for tag in tags)  # Get the maximum probability and corresponding state
    return path[state], prob  # Return the best sequence and its probability


def emit(tag, word, emit_probs, tag_totals, n_tags):
    """
    Function to calculate the emission probability with smoothing.

    :param tag: Current state
    :param word: Current observation
    :param emit_probs: Dictionary of emission probabilities
    :param tag_totals: Total counts of occurrences for each state
    :param n_tags: Total number of states
    :return: Emission probability for the given observation and state
    """
    return emit_probs.get(tag, {}).get(word, 1 / (sum(tag_totals.values()) + n_tags))  # Apply smoothing


def parse_conllu(file_path):
    """
    Function to parse a CoNLL-U file into a list of sentences with words and POS tags.

    :param file_path: Path to the CoNLL-U file
    :return: List of tuples containing words and their corresponding POS tags
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = parse(f.read())  # Parse the file contents
    return [
        ([token['form'] for token in sent if 'form' in token],
         [token['upos'] for token in sent if 'upos' in token])
        for sent in sentences
    ]


def demo():
    """
    Function to demonstrate the Viterbi algorithm with an example.
    """
    # Define example HMM parameters
    start_probs = {'HOT': 0.8, 'COLD': 0.2}
    trans_probs = {
        'HOT': {'HOT': 0.7, 'COLD': 0.3},
        'COLD': {'HOT': 0.4, 'COLD': 0.6}
    }
    emit_probs = {
        'HOT': {1: 0.2, 2: 0.4, 3: 0.4},
        'COLD': {1: 0.5, 2: 0.4, 3: 0.1}
    }
    tag_totals = {'HOT': 3, 'COLD': 3}
    obs = [3, 1, 3]  # Observation sequence

    # Execute the Viterbi algorithm
    start_time = time.time()  # Record the start time
    best_seq, prob = viterbi(obs, start_probs, trans_probs, emit_probs, tag_totals)
    end_time = time.time()  # Record the end time

    # Display results
    print("Best sequence:", best_seq)
    print("Probability:", f"{prob:.6f}")
    print("Time taken:", f"{end_time - start_time:.6f} seconds")


if __name__ == '__main__':
    demo()
