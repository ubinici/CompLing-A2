# README

## **Author**

Ümit Altar Binici

## **Directory Structure**

```plaintext
|
├── hmm_core.py          # Core functionality for the HMM, including Viterbi algorithm.
├── hmm_train.py         # Script for training the HMM and estimating parameters.
├── hmm_eval.py          # Script for evaluating the HMM's performance on test data.
├── hmm_curve.py         # Script for generating a learning curve by varying training sizes.
├── start_probs.csv      # Output CSV file containing starting probabilities.
├── trans_probs.csv      # Output CSV file containing transition probabilities.
├── emit_probs.csv       # Output CSV file containing emission probabilities.
├── de_gsd-ud-train.conllu  # Training dataset (CoNLL-U formatted).
├── de_gsd-ud-test.conllu   # Test dataset (CoNLL-U formatted).
└── README.md            # This README file.
```

## **Versions**

- **Python:** 3.11.5  
- **Libraries:**  
  - `conllu`: 4.4.2  
  - `matplotlib`: 3.9.2  
  - `collections`: Standard library module (no version required).

## **Runtime**

Approximate runtimes for each script on a typical machine:

1. **`hmm_train.py`:**  
   - **Runtime:** ~30 seconds for estimating parameters from `de_gsd-ud-train.conllu`.  
   - Note: Runtime scales linearly with dataset size.  

2. **`hmm_eval.py`:**  
   - **Runtime:** ~15 seconds for predictions and evaluation on `de_gsd-ud-test.conllu`.  

3. **`hmm_curve.py`:**  
   - **Runtime:** Varies depending on `step_size` (default: ~2–5 minutes for 10 steps with `step_size=500`).

## **Additional Features**

- **Learning Curve Visualization (`hmm_curve.py`)**  
  - **Feature:** Generates a plot showing accuracy as a function of training size.  
  - **Implementation:** Combines incremental training and evaluation to assess model improvement.

- **Error Diagnostics (`hmm_eval.py`)**  
  - **Feature:** Displays most common prediction errors for error analysis.  
  - **Implementation:** `display_diagnostics` function.

## **External Material**

- **Dataset:**  
  - **German GSD Universal Dependencies corpus**:  
    - [Training Set](https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/refs/heads/master/de_gsd-ud-train.conllu)  
    - [Test Set](https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/refs/heads/master/de_gsd-ud-test.conllu)

- **Python Libraries:**  
  - **`conllu`**: For parsing CoNLL-U formatted data.  
  - **`matplotlib`**: For plotting learning curves.
