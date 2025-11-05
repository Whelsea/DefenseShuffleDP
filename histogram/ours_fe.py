import math
import random
import numpy as np
from itertools import chain
import bisect
from typing import List, Tuple, Dict, Any
import os
import time


try:
    import FE1

    FE1Baseline = FE1.FE1Baseline
except ImportError:
    print("Error: FE1.py module not found. Please ensure it is in the path.")

    class DummyFE1:
        def __init__(self, *args, **kwargs): self.n, self.B, self.sample_prob = 1, 1, 0

        def local_randomizer(self, x): return [(1, 1, 1)]

        def analyzer(self, msgs): return np.zeros(self.B + 1)

        def get_theta_fe1(self, beta): return 100.0


    FE1Baseline = DummyFE1

# # =================== Default parameter setting (global) ===================
num_users = 4096
domain = 2
epsilon = 1.0
delta = 1.0 / num_users / num_users
k = 0
times = 1
real_sums = []
sorted_malicious = []
malicious_users = []  # Set of attacker indices
beta = 0.1  # Confidence level for HSDP detection (1-beta)
d = 2 ** 24  # Domain size B
C = 1.0  # Parameter c for FE1's b calculation (b=n/log^c n)
custom_lambda_n = None
workers = 4  # Number of workers for FE1's multi-process analyzer


# ------------------------------------------------------------
#  Determine lambda based on n
# ------------------------------------------------------------
def find_lambda(n):
    if custom_lambda_n is not None:
        return custom_lambda_n
    if n < 1:
        return 0
    # Use the formula log2(n) * log2(1/delta) for lambda base
    the_lam = math.log2(n) * math.log2(1 / delta)

    # Round up to the next power of 2, then scale by 4 (following the original provided logic)
    exponent = math.ceil(math.log2(the_lam))
    lam_power_of_2 = 2 ** exponent
    return int(lam_power_of_2 * 4)


# ------------------------------------------------------------
#  Attacker Message Generator
# ------------------------------------------------------------
def attacker_messages(num_messages: int) -> List[Tuple[int, int, int]]:
    """
    Simulates the messages sent by a poisoning attacker.
    For simplicity and following typical attack vector length, we send placeholder messages.
    A sophisticated attack would tailor these based on FE1 parameters.
    """
    # Placeholder attack vector: n messages of (1, 1, 1) tuple
    return [(1, 1, 1)] * num_messages


# ------------------------------------------------------------
#  Initialize FE1 Objects for all layers (HSDP budget split)
# ------------------------------------------------------------
def init_FE1(n: int, d: int, L: int) -> List[FE1Baseline]:
    """Initializes FE1Baseline objects for each layer with split privacy budget."""

    eps_part1 = epsilon / 2 / (L - 1)
    eps_part2 = epsilon / 2
    delta_part1 = delta / (2 * (L - 1))
    delta_part2 = delta / 2
    beta_part1 = beta / 2 / (2 ** L - 2)  # For lower layers (r=0 to L-2)
    beta_part2 = beta / 2  # For the top layer (r=L-1)


    fe1_list = []

    for r in range(L):
        group_size = find_lambda(n) * (2 ** r)

        privacy_scale = max((2 ** r - 1) / (2 ** r), 1)

        # Determine current layer's privacy budget (eps, delta)
        if r == L - 1:  # Top layer
            current_n = n
            current_eps = eps_part2 * (n - 1) / n
            current_delta = delta_part2 * (n - 1) / n
            current_beta = beta_part2  # Use beta_part2 for the top layer
        else:  # Lower layers
            current_n = group_size
            current_eps = eps_part1 * privacy_scale
            current_delta = delta_part1 * privacy_scale
            current_beta = beta_part1  # Use beta_part1 for lower layers

        fe1_r = FE1Baseline(n=current_n, B=d,
                            epsilon=current_eps,
                            delta=current_delta,
                            c=C, use_mu_search=False,
                            beta=current_beta) 
        fe1_list.append(fe1_r)

    return fe1_list

# ------------------------------------------------------------
#  2) Backtrack Function for FE1 (Recovery)
# ------------------------------------------------------------
def backtrack_fe1(max_r, Q, B):
    """
    Recovers the frequency vector by propagating valid counts upwards.
    Q[r][g][1] == float('-inf') is used as the invalid marker.
    """
    # print("backtrack")
    group_count_0 = Q[0].shape[0]

    # Base case (r=0) recovery: Set invalid leaf nodes to zero vector
    for g in range(group_count_0):
        # Check invalid flag at index 1
        if Q[0][g][1] == float('-inf'):
            Q[0][g] = np.zeros(B + 1)

    # Hierarchical recovery: Fill invalid parent nodes with sum of (recovered) children
    for r in range(1, max_r + 1):
        group_count_r = Q[r].shape[0]
        for g in range(group_count_r):
            # Check invalid flag at index 1
            if Q[r][g][1] == float('-inf'):
                left = 2 * g
                right = 2 * g + 1

                # Check bounds for children arrays
                if left < Q[r - 1].shape[0] and right < Q[r - 1].shape[0]:
                    # Sum of (potentially already recovered) children vectors
                    Q[r][g] = Q[r - 1][left] + Q[r - 1][right]

    return Q[max_r][0]  # final recovered frequency vector


# ------------------------------------------------------------
#  3) Analyzer of HSDP-FE1 (Detection + Recovery)
# ------------------------------------------------------------
def analyzer_fe1(Q: List[np.ndarray], n: int, fe1_list: List[FE1Baseline]):
    """
    Performs hierarchical detection and recovery on the aggregated results Q.
    Q: List[np.ndarray], shape of each Q[r] is (group_count[r], B+1)
    """
    lambda_n = find_lambda(n)
    L = len(fe1_list)  # L = int(math.ceil(math.log2(n / lambda_n))) + 1
    B = fe1_list[0].B

    if k == 0:
        return Q[L - 1][0]  # No attackers, no detection needed

    # 1. Calculate detection thresholds (theta) for all layers
    theta_list = []
    # HSDP beta split logic (assuming total split over L-1 lower layers + 1 top layer)
    beta_part1 = beta / (2 * (2 * n / lambda_n - 2))
    beta_part2 = beta / 2

    for r in range(L):
        # Assumes FE1Baseline has a method to get L-inf bound (theta) based on its initialized parameters and confidence beta
        theta_list.append(fe1_list[r].get_theta_fe1())

        # 2. Bottom-layer Detection (r=0)
    group_count_0 = Q[0].shape[0]
    theta_0 = theta_list[0]
    N_0 = fe1_list[0].n  # Group size = lambda_n

    for g in range(group_count_0):
        val = Q[0][g][1:]  # Frequency counts for bins [1..B]

        # Check if any frequency count is outside the valid range [-theta, N + theta]
        if np.max(val) > (N_0 + theta_0) or np.min(val) < -theta_0:
            # Mark this group invalid: use the whole vector as marker, and set [1] as the inf flag
            Q[0][g] = np.array([float('-inf')] * (B + 1))
            Q[0][g][1] = float('-inf')

    # 3. Hierarchical Detection (r > 0)
    for r in range(1, L):
        group_count_r = Q[r].shape[0]
        theta_r = theta_list[r]
        theta_prev = theta_list[r - 1]

        for g in range(group_count_r):
            left_idx = 2 * g
            right_idx = 2 * g + 1

            # If any subgroup is invalid, mark parent as invalid
            if (Q[r - 1][left_idx][1] == float('-inf') or
                    Q[r - 1][right_idx][1] == float('-inf')):
                Q[r][g] = np.array([float('-inf')] * (B + 1))
                Q[r][g][1] = float('-inf')
                continue

            # Compare current group result with sum of subgroups (L-inf norm of the difference vector)
            diff = np.max(np.abs(Q[r][g][1:] - Q[r - 1][left_idx][1:] - Q[r - 1][right_idx][1:]))

            # Detection threshold: 2 * theta_prev + theta_r
            if diff > 2 * theta_prev + theta_r:
                Q[r][g] = np.array([float('-inf')] * (B + 1))
                Q[r][g][1] = float('-inf')

    # 4. Backtrack for Recovery
    A = backtrack_fe1(L - 1, Q, B)
    return A


# ------------------------------------------------------------
#  4) Main HSDP-FE1 Protocol Execution
# ------------------------------------------------------------
def ours_FE1(values: List[int]):
    """
    Executes the HSDP-FE1 protocol (Randomization, Aggregation, Analyzer).
    Values are assumed to be 0-indexed [0, d-1]. FE1 expects 1-indexed [1, d].
    """
    n = len(values)
    B = d  # Global domain size
    lambda_n = find_lambda(n)
    L = int(math.ceil(math.log2(n / lambda_n))) + 1

    # 1. Initialize FE1Baseline objects for all layers
    fe1_list = init_FE1(n, B, L)

    # 2. Randomize all users for all layers
    all_messages = [[] for _ in range(n)]  # all_messages[user_idx][layer_idx] = List[msgs]

    # Calculate max messages for attacker to mimic (using top layer's max)
    max_msgs_per_user = fe1_list[L - 1].send_fixed_messages + 1
    total_honest_messages = 0

    for i in range(n):
        # FE1 expects 1-indexed values [1, B]
        v_fe1 = values[i] + 1

        # Check if user i is an attacker
        if i in malicious_users:
            # Attacker messages (mimic the structure and length of max honest user)
            m_r = [attacker_messages(n) for _ in range(L)]
        else:
            # Honest user randomization
            m_r = [fe1.local_randomizer(v_fe1) for fe1 in fe1_list]

            # Track honest messages sent across all layers
            for fe1, msgs in zip(fe1_list, m_r):
                total_honest_messages += len(msgs)

        all_messages[i] = m_r

    # 3. Aggregate messages into groups and run Analyzer
    Q = [np.zeros((int(math.ceil(n / (lambda_n * (2 ** r)))), B + 1)) for r in range(L)]

    for r in range(L):
        group_size = lambda_n * (2 ** r)
        fe1_r = fe1_list[r]

        for g in range(len(Q[r])):
            start_idx = g * group_size
            end_idx = min((g + 1) * group_size, n)

            # Collect all messages for this group in this layer
            group_messages = []
            for i in range(start_idx, end_idx):
                group_messages.extend(all_messages[i][r])

                # Run the FE1 Analyzer on the collected messages
            est_freq = fe1_r.analyzer(group_messages)

            # Store the estimated frequency vector in Q[r][g]
            Q[r][g] = est_freq

    # Total messages per honest user
    nmessages_per_user = total_honest_messages / max(1, n - k)

    # 4. HSDP Analyzer: Detection + Recovery
    dp_freqvec = analyzer_fe1(Q, n, fe1_list)

    return dp_freqvec, nmessages_per_user


# ------------------------------------------------------------
#  Data Loading and Main Experiment Logic
# ------------------------------------------------------------

def loaddata(path: str) -> List[int]:
    """Loads data, assumes 0-indexed values for consistency with FE1 input preparation."""
    with open(path, 'r') as f:
        n = int(f.readline())
        d = int(f.readline())
        # Data values are assumed to be 0-indexed [0, B-1]
        return [int(f.readline()) for _ in range(n)], n, d


def load_data_by_mode(data_mode: str, n: int, B: int):
    """Loads or simulates data based on mode."""
    # Assuming the data structure expects 0-indexed values for simplicity
    if data_mode == "zipf":
        path = f"./data/Zip/Zip_n{n}B{B}"
        data, _, _ = loaddata(path)
    elif data_mode == "gauss":
        path = f"./data/Gauss/Gauss_n{n}B{B}"
        data, _, _ = loaddata(path)
    elif data_mode == "aol" or data_mode == "twitter":
        # Placeholder path for real-world data
        path = f"./data/Real/{data_mode}_n{n}B{B}.txt"
        data, _, _ = loaddata(path)
    elif data_mode == "unif":
        # Generate 0-indexed values [0, B-1]
        data = np.random.randint(0, B, size=n).tolist()
    else:
        raise ValueError(f"Unsupported data_mode: {data_mode}")
    return data


if __name__ == "__main__":
    # === Global Parameters from the user's script ===
    epsilon = 4.0
    C = 1.0  # Use capital C for the global variable
    k = 1  # Set k to 1 for initial test
    times = 10
    beta = 0.1

    list_distribution = ["Unif"]  # Simplified list
    dataset_mode = "simulated dataset"

    # === Experiment Loops ===
    # Using the exact n, d from user's original script example
    for n in {131072}:
        for d_size in {131072}:
            for k_attack in {1}:
                # Update global parameters for the current loop run
                num_users = n
                d = d_size
                k = k_attack
                delta = 1.0 / (n ** 2)

                for distribution in list_distribution:
                    # 1. Data Setup
                    data_mode = distribution.lower()
                    values = load_data_by_mode(data_mode, n, d)

                    malicious_users = set(random.sample(range(n), k)) if k > 0 else set()
                    sorted_malicious = sorted(malicious_users)

                    # 2. Run Experiments and Collect Errors
                    max_errors = []
                    l50Errors = []
                    l90Errors = []
                    l95Errors = []
                    l99Errors = []

                    current_nmessages_per_user = 0

                    for t in range(times):
                        print(f"Running n={n}, B={d}, k={k}, dist={distribution}, time={t}")

                        est_freq, nmessages_per_user = ours_FE1(values)
                        current_nmessages_per_user = nmessages_per_user  # Store the last message count

                        # True frequency calculation (1-indexed)
                        true_freq = np.zeros(d + 1)
                        for x in values:
                            # Values are 0-indexed, FE1 analysis is 1-indexed
                            true_freq[x + 1] += 1

                            # Error evaluation (L-inf norm: max error)
                        # Only compare bins [1..d]
                        errors = np.abs(true_freq[1:] - est_freq[1:])
                        max_error = np.max(errors)
                        max_errors.append(max_error)

                        sorted_errors = sorted(errors)
                        l50Errors.append(sorted_errors[int(0.50 * d)])
                        l90Errors.append(sorted_errors[int(0.90 * d)])
                        l95Errors.append(sorted_errors[int(0.95 * d)])
                        l99Errors.append(sorted_errors[int(0.99 * d)])

                    # 3. Trimming and Averaging
                    cut_times = int(0.1 * times)


                    def trimmed_mean(data, cut):
                        return np.mean(sorted(data)[cut:-cut]) if cut > 0 else np.mean(data)


                    max_error_avg = trimmed_mean(max_errors, cut_times)
                    l50Error_avg = trimmed_mean(l50Errors, cut_times)
                    l90Error_avg = trimmed_mean(l90Errors, cut_times)
                    l95Error_avg = trimmed_mean(l95Errors, cut_times)
                    l99Error_avg = trimmed_mean(l99Errors, cut_times)

                    # 4. Write Results to File
                    result_root = os.path.join(".", "Result")
                    baseline_folder = os.path.join(result_root, "ours+FE1_full")
                    os.makedirs(baseline_folder, exist_ok=True)
                    data_folder = os.path.join(baseline_folder, f"{distribution}")
                    os.makedirs(data_folder, exist_ok=True)

                    outfile_name = f"ours+FE1_n{n}_B{d}_attacker{k}_lambda{find_lambda(n)}.txt"
                    outfile_path = os.path.join(data_folder, outfile_name)

                    with open(outfile_path, 'w') as f:
                        f.write(f"lambda: {find_lambda(n)}\n")
                        f.write(f"L-inf error: {max_error_avg}\n")
                        f.write(f"l50error: {l50Error_avg}\n")
                        f.write(f"l90error: {l90Error_avg}\n")
                        f.write(f"l95error: {l95Error_avg}\n")
                        f.write(f"l99error: {l99Error_avg}\n")
                        f.write(f"#messages per user: {current_nmessages_per_user}\n")

                    print(f"Completed n={n}, B={d}, k={k}, dist={distribution}. Results saved.")
