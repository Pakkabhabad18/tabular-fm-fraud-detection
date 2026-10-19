"""Fixed settings shared by every experiment.

Anything that changes how results are computed lives here, so a change is one
edit in one place and shows up clearly in the git history.
"""

# Five repeats, as in the proposal. Each seed gives a different stratified split
# and a different random draw of negatives, so the spread across seeds shows how
# much a result moves just from which rows were picked.
SEEDS = [0, 1, 2, 3, 4]

# Train / validation / test fractions. The proposal fixes a stratified split but
# gives no ratios; 60/20/20 is recorded in the decision log.
SPLIT_FRACTIONS = (0.60, 0.20, 0.20)

# Fraud rates (pi_1) in the prevalence sweep, highest first. "natural" means the
# untouched ULB rate of 492 / 284,807 = 0.172%. The others are made by
# downsampling legitimate rows inside each split (frauds are never dropped).
PREVALENCE_LEVELS = {
    "20": 0.20,
    "5": 0.05,
    "1": 0.01,
    "0.5": 0.005,
    "natural": None,
}

# Context budgets k for the foundation models (task T5 onward).
CONTEXT_BUDGETS = [1_000, 2_000, 5_000, 10_000]

# Where large files go when running on Colab. Everything here stays out of git.
DRIVE_DIR = "/content/drive/MyDrive/tabular-fm-fraud-detection"
