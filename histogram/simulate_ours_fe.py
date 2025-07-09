import math
import random
import numpy as np
import pandas as pd
from itertools import chain
import bisect
import FE1_Simulator
from typing import List
import os

'''
    Can set lambda, n and lambda need to be powers of 2
    Attacker behavior: send n messages with value u
'''
# # =================== Default parameter setting (global) ===================
num_users = 4096
domain = 2
epsilon = 1.0
delta = 1.0 / num_users / num_users
k = 0
times = 1
real_sums = []
sorted_malicious = []
malicious_users = []  # for fe
beta = 0.1
d = 2 ** 24
C = 1.0
custom_lambda_n = None


# ------------------------------------------------------------
#  Determine lambda based on n
#   Input: n
#   Output: lambda (log^2 n)
#       ** lambda is the largest integer satisfying:
#          1) ∈ [log^2 n, log^2.5 n]
#          2) Is a power of 2
#
#   Currently used for bit counting/summation/frequency estimation problems
# ------------------------------------------------------------
def find_lambda(n):
    if custom_lambda_n is not None:
        return custom_lambda_n
    if n < 1:
        return 0
    the_lam = math.log2(n) * math.log2(1 / delta)

    exponent = math.ceil(math.log2(the_lam))
    lam_power_of_2 = 2 ** exponent
    print(f"n={n},lambda={the_lam}, real_lambda={lam_power_of_2 * 4}")
    return int(lam_power_of_2 * 4)


def simulate_analyzer_fe1(Q, n, fe1_list):
    """
    Q: List[np.ndarray], shape of each Q[r] is (group_count[r], B+1)
    L: total number of layers
    lambda_n: leaf group size
    theta: scalar threshold for detection (based on Linf error)
    B: domain size
    """

    lambda_n = find_lambda(n)
    L = int(math.ceil(math.log2(n / lambda_n))) + 1
    if k == 0:
        return Q[L - 1][0]  # top-level frequency vector

    # ========== (1) Bottom-layer Detection (r=0) ==========
    group_count_0 = Q[0].shape[0]

    theta_list = []
    for r in range(L):
        theta_list.append(fe1_list[r].get_theta_fe1())

    for g in range(group_count_0):
        val = Q[0][g]
        if np.max(val) > (lambda_n + theta_list[0]) or np.min(val) < -theta_list[0]:
            print("error")
            print("val: ", val, "theta: ", theta_list[0])
            # mark this group invalid
            Q[0][g][1] = float('-inf')
    # ========== (2) Hierarchical Detection ==========
    for r in range(1, L):
        group_count_r = Q[r].shape[0]
        theta_r = fe1_list[r].get_theta_fe1()
        for g in range(group_count_r):
            left_idx = 2 * g
            right_idx = 2 * g + 1
            if (Q[r - 1][left_idx][1] == float('-inf') or
                    Q[r - 1][right_idx][1] == float('-inf')):
                Q[r][g][1] = float('-inf')
                continue
            diff = np.max(np.abs(Q[r][g] - Q[r - 1][left_idx] - Q[r - 1][right_idx]))

            if diff > 2 * theta_list[r - 1] + theta_list[r]:
                print("error")
                print("diff: ", diff, "theta: ", 2 * theta_list[r - 1] + theta_list[r])
                Q[r][g][1] = float('-inf') * np.ones(d + 1)

    # ========== (3) Backtrack for Recovery ==========
    A = backtrack_fe1(L - 1, Q, d)
    return A


def backtrack_fe1(max_r, Q, d):
    print("backtrack")
    group_count = Q[0].shape[0]
    for g in range(group_count):
        if Q[0][g][1] == float('-inf'):
            Q[0][g] = np.zeros(d + 1)

    for r in range(1, max_r + 1):
        group_count_r = Q[r].shape[0]
        for g in range(group_count_r):
            if Q[r][g][1] == float('-inf'):
                left = 2 * g
                right = 2 * g + 1
                if left < Q[r - 1].shape[0] and right < Q[r - 1].shape[0]:
                    Q[r][g] = Q[r - 1][left] + Q[r - 1][right]

    return Q[max_r][0]  # final recovered frequency vector


def simulate_ours_FE1(values):
    n = len(values)
    lambda_n = find_lambda(n)
    print(lambda_n)
    L = int(math.ceil(math.log2(n / lambda_n))) + 1

    eps_part1 = epsilon / 2 / (L - 1)
    eps_part2 = epsilon / 2
    delta_part1 = delta / (2 * (L - 1))  # split privacy budget
    delta_part2 = delta / 2
    # beta_part = beta / (2 * n / lambda_n - 1)
    beta_part1 = beta / (2 * (2 * n / lambda_n - 2))
    beta_part2 = beta / 2
    # delta_part = delta / L

    # initial fe1 objects
    fe1_list = []
    for r in range(L - 1):
        privacy_scale = max((2 ** r - 1) / (2 ** r), 1)
        fe1_r = FE1_Simulator.FE1Simulator(n=lambda_n * (2 ** r), B=d, epsilon=eps_part1 * privacy_scale,
                                           delta=delta_part1 * privacy_scale,
                                           c=C, beta=beta_part1)
        fe1_list.append(fe1_r)
    fe1_r = FE1_Simulator.FE1Simulator(n=n, B=d, epsilon=eps_part2 * (num_users - 1) / num_users,
                                       delta=delta_part2 * (num_users - 1) / num_users,
                                       c=C, beta=beta_part2)
    fe1_list.append(fe1_r)

    print("start running")
    Q = [np.zeros((int(math.ceil(n / (lambda_n * (2 ** r)))), d + 1)) for r in range(L)]
    total_messages = 0

    for r in range(L):
        group_size = lambda_n * (2 ** r)
        print(f"r={r}")
        for g in range(len(Q[r])):
            start_idx = g * group_size
            end_idx = min((g + 1) * group_size, n)

            # Calculate number of attackers in current group
            left = bisect.bisect_left(sorted_malicious, start_idx)
            right = bisect.bisect_right(sorted_malicious, end_idx)

            attackers_count_g = right - left
            honest_user_proportion = (group_size - attackers_count_g) / group_size

            # freq vec for honest users
            group_values = [values[i] for i in range(start_idx, end_idx)]
            honest_user_values = [v for idx, v in enumerate(group_values)
                                  if (start_idx + idx) not in malicious_users]

            est_huser_freq = fe1_list[r].simulate_parallel(honest_user_values, honest_user_proportion)
            print(r)
            # freq vec for poisoning attacker
            noisy_attacker_freq = np.zeros(d + 1)
            for _ in range(attackers_count_g):
                # noisy_bin = np.random.randint(0, d)
                # noisy_attacker_freq[noisy_bin] += n
                for noisy_bin in range(0, d):
                    noisy_attacker_freq[noisy_bin] += n

            Q[r][g] = est_huser_freq + noisy_attacker_freq
        print("comp")
        total_messages += 1 + fe1_list[r].sample_prob

    dp_freqvec = simulate_analyzer_fe1(Q, n, fe1_list)

    return dp_freqvec, total_messages


def loaddata(path: str) -> List[int]:
    with open(path, 'r') as f:
        n = int(f.readline())
        d = int(f.readline())
        return [int(f.readline()) for _ in range(n)], n, d


if __name__ == "__main__":
    epsilon = 4.0
    c = 1.0
    k = 0
    times = 50
    beta = 0.1
    # list_distribution = ["Unif", "Gauss", "Zip"]
    # list_distribution = ["Unif"]
    list_distribution = ["Unif", "Zip"]
    # dataset = "realworld dataset"
    dataset = "simulated dataset"
    # for n in {16384, 131072, 1048576, 16777216}:
    #     for d in {1048576, 16777216}:

    for n in {131072}:
        for d in {131072}:
            for k in {0, 1}:
                if dataset == "realworld dataset":
                    for dataset_name in {"twitter"}:
                        if dataset_name == "aol":
                            b = math.log2(d)
                            values, n, d = loaddata(f"./data/aol_data/trans_01_n_{n}_b_{int(b)}_B_{d}.txt")
                        else:
                            values, n, d = loaddata(f"./data/twitter_data/twitwer_n{131072}B{d}.txt")
                        delta = 1.0 / (n ** 2)

                        malicious_users = []
                        if k > 0:
                            malicious_users = set(random.sample(range(n), k))
                        sorted_malicious = sorted(malicious_users)

                        max_errors = []
                        l50Errors = []
                        l90Errors = []
                        l95Errors = []
                        l99Errors = []

                        for t in range(times):
                            print(f"running n={n}, B={d}, time={t}")
                            est_freq, nmessages_per_user = simulate_ours_FE1(values)

                            true_freq = np.zeros(d + 1)
                            for x in values:
                                true_freq[x] += 1

                            errors = np.abs(true_freq[1:] - est_freq[1:])
                            max_error = np.max(errors)
                            max_errors.append(max_error)
                            max_error_index = np.argmax(errors) + 1
                            sorted_errors = sorted(errors)
                            l50Errors.append(sorted_errors[int(0.50 * d)])
                            l90Errors.append(sorted_errors[int(0.90 * d)])
                            l95Errors.append(sorted_errors[int(0.95 * d)])
                            l99Errors.append(sorted_errors[int(0.99 * d)])

                        cut_times = int(0.1 * times)
                        max_error = np.mean(sorted(max_errors)[cut_times:-cut_times])
                        l50Error = np.mean(sorted(l50Errors)[cut_times:-cut_times])
                        l90Error = np.mean(sorted(l90Errors)[cut_times:-cut_times])
                        l95Error = np.mean(sorted(l95Errors)[cut_times:-cut_times])
                        l99Error = np.mean(sorted(l99Errors)[cut_times:-cut_times])

                        # Write results to file
                        result_root = os.path.join(".", "Result")
                        baseline_folder = os.path.join(result_root, "ours+FE1")
                        if not os.path.exists(baseline_folder):
                            os.makedirs(baseline_folder)
                        if dataset == "realworld dataset":
                            data_folder = os.path.join(baseline_folder, f"{dataset_name}")
                        else:
                            data_folder = os.path.join(baseline_folder, f"{distribution}")
                        if not os.path.exists(data_folder):
                            os.makedirs(data_folder)
                        outfile_name = f"ours+FE1_n{n}_B{d}_attacker{k}.txt"
                        outfile_path = os.path.join(data_folder, outfile_name)

                        with open(outfile_path, 'w', encoding='utf-8') as f:
                            # f.write("linf-errors:" + max_errors + "\n")
                            f.write("L-inf error: " + str(max_error) + "\n")
                            f.write("l50error: " + str(l50Error) + "\n")
                            f.write("l90error: " + str(l90Error) + "\n")
                            f.write("l95error: " + str(l95Error) + "\n")
                            f.write("l99error: " + str(l99Error) + "\n")
                            f.write("#messages per user: " + str(nmessages_per_user) + "\n")
                            # f.write("#Bits per user: "+ str(math.ceil(math.log2())))
                        print(f"comp n={n} B={d}")
                else:
                    for distribution in list_distribution:
                        if distribution == "Unif":
                            values = np.random.randint(0, d, size=n)
                        else:
                            values, n, d = loaddata(f"./data/{distribution}/{distribution}_n{n}B{d}")
                        delta = 1.0 / (n ** 2)

                        malicious_users = []
                        if k > 0:
                            malicious_users = set(random.sample(range(n), k))
                        sorted_malicious = sorted(malicious_users)

                        max_errors = []
                        l50Errors = []
                        l90Errors = []
                        l95Errors = []
                        l99Errors = []

                        for t in range(times):
                            print(f"running n={n}, B={d}, time={t}")
                            est_freq, nmessages_per_user = simulate_ours_FE1(values)

                            # 误差评估
                            true_freq = np.zeros(d + 1)
                            for x in values:
                                true_freq[x] += 1

                            errors = np.abs(true_freq[1:] - est_freq[1:])
                            max_error = np.max(errors)
                            print("error=", max_error)
                            max_errors.append(max_error)
                            max_error_index = np.argmax(errors) + 1
                            sorted_errors = sorted(errors)
                            l50Errors.append(sorted_errors[int(0.50 * d)])
                            l90Errors.append(sorted_errors[int(0.90 * d)])
                            l95Errors.append(sorted_errors[int(0.95 * d)])
                            l99Errors.append(sorted_errors[int(0.99 * d)])

                        cut_times = int(0.1 * times)
                        max_error = np.mean(sorted(max_errors)[cut_times:-cut_times])
                        l50Error = np.mean(sorted(l50Errors)[cut_times:-cut_times])
                        l90Error = np.mean(sorted(l90Errors)[cut_times:-cut_times])
                        l95Error = np.mean(sorted(l95Errors)[cut_times:-cut_times])
                        l99Error = np.mean(sorted(l99Errors)[cut_times:-cut_times])

                        # Write results to file
                        result_root = os.path.join(".", "Result")
                        baseline_folder = os.path.join(result_root, "ours+FE1")
                        if not os.path.exists(baseline_folder):
                            os.makedirs(baseline_folder)
                        if dataset == "aol dataset":
                            data_folder = os.path.join(baseline_folder, "aol_data")
                        else:
                            data_folder = os.path.join(baseline_folder, f"{distribution}")
                        if not os.path.exists(data_folder):
                            os.makedirs(data_folder)
                        outfile_name = f"ours+FE1_n{n}_B{d}_attacker{k}_lambda{find_lambda(n)}_revise.txt"
                        outfile_path = os.path.join(data_folder, outfile_name)

                        with open(outfile_path, 'w', encoding='utf-8') as f:
                            f.write("lambda: " + str(find_lambda(n)))
                            f.write("L-inf error: " + str(max_error) + "\n")
                            f.write("l50error: " + str(l50Error) + "\n")
                            f.write("l90error: " + str(l90Error) + "\n")
                            f.write("l95error: " + str(l95Error) + "\n")
                            f.write("l99error: " + str(l99Error) + "\n")
                            f.write("#messages per user: " + str(nmessages_per_user) + "\n")
                            # f.write("#Bits per user: "+ str(math.ceil(math.log2())))
                        print(f"comp n={n} B={d}")
