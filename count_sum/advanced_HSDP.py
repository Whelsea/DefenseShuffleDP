import math
import BBGN
import GKMPS
import numpy as np
from itertools import chain
import bisect
from typing import List

# # =================== default parameter setting (global)===================
num_users = 4096
domain = 2
epsilon = 1.0
delta = 1.0 / num_users / num_users
k = 0
times = 1
real_sums = []
sorted_malicious = [[]]
malicious_users = []
custom_lambda_n = None
# ============== GKMPS =============
gamma = 0.3
beta = 0.1
# ============== BBGN =============
sigma = 40
# ============== FE1 =============
B = 2 ** 24
C = 1.0


# ------------------------------------------------------------
#   Set lambda
#   Input: n
#   Output: logn * log(1/delta) * c
# ------------------------------------------------------------
def find_lambda(n):
    if custom_lambda_n is not None:
        return custom_lambda_n
    if n < 1:
        return 0
    the_lam = math.log2(n) * math.log2(1 / delta)

    exponent = math.ceil(math.log2(the_lam))
    lam_power_of_2 = 2 ** exponent
    return int(lam_power_of_2 / 8)


# ------------------------------------------------------------
#   Set reasonable error bounds of baseline protocols
# ------------------------------------------------------------
def get_theta(baseline, beta):
    if baseline.name == "GKMPS":
        theta = (baseline.domain / baseline.epsilonstar) * math.log(
            (2 * math.exp(baseline.epsilonstar)) / (beta * (math.exp(baseline.epsilonstar) + 1)))
    if baseline.name == "BBGN":
        theta = (baseline.U / baseline.epsilon) * math.log(
            (2 * math.exp(baseline.epsilon)) / (beta * (math.exp(baseline.epsilon) + 1)))
    return theta


# ------------------------------------------------------------
#  1) Randomizer of HSDP
#   Input:
#       - gkmps_list[r] : GKMPS object used in layer r
#       - value (the user hold)
#   Output: messages[r][j] (multiple lists of messages in each layer r)
#
# ------------------------------------------------------------
def LocalRandomizer(baseline, value):
    messages = []
    for baseline_r in baseline:
        m = baseline_r.LocalRandomizer(value)
        messages.append(m)
    return messages


# ------------------------------------------------------------
#  2) Backtrack Function
#   Input:
#       -r: current layer
#       -g: current group
#       -Q: multiple lists that store aggregated results for all groups at all layers.
#   Output: the valid result of group g in layer r after recovery.
# ------------------------------------------------------------
def Backtrack(max_r, Q):
    group_count = len(Q[0])
    for g in range(group_count):
        if Q[0][g] == float('-inf'):
            Q[0][g] = 0
    for r in range(1, max_r + 1):
        group_count = len(Q[r])
        for g in range(group_count):
            if Q[r][g] == float('-inf'):
                left = 2 * g
                right = 2 * g + 1
                if left < len(Q[r - 1]) and right < len(Q[r - 1]) and r > 0:
                    Q[r][g] = Q[r - 1][left] + Q[r - 1][right]
    return Q[max_r][0]


# ------------------------------------------------------------
#  3) Analyzer of HSDP
#   Input:
#       - gkmps_list: list of GKMPS objects
#       - all_messages[i][r]: three-dimensional array of all messages sent by n users at all layers
#       - n: number of users
#       - theta: error threshold of single group
#   Output: A (final aggregated result)
#
# ------------------------------------------------------------
def Analyzer(baseline, all_messages):
    lambda_n = find_lambda(num_users)
    L = int(math.ceil(math.log2(math.ceil(num_users / lambda_n)))) + 1
    max_r = L - 1
    Q = [[] for _ in range(L)]
    eps_part1 = epsilon / 2 / (L - 1)  # split privacy budget
    eps_part2 = epsilon / 2
    delta_part1 = delta / 2 / (L - 1)
    delta_part2 = delta / 2
    # beta_part = beta / (2 * num_users / lambda_n - 1)
    beta_part1 = beta / 2 / (2 ** L - 2)
    beta_part2 = beta / 2

    # ========== (1) Bottom-layer Detection ( r=0 ) ==========

    Q[0] = [0.0] * math.ceil(num_users / lambda_n)
    baseline_0 = baseline[0]
    theta = get_theta(baseline_0, beta_part1)
    group_count = math.ceil(num_users / lambda_n)
    for g in range(group_count):
        # compute result of gourp g
        start_idx = g * lambda_n
        end_idx = min((g + 1) * lambda_n, num_users)
        group_messages = [all_messages[i][0] for i in range(start_idx, end_idx)]
        flattened_group_messages = list(chain.from_iterable(group_messages))
        result = baseline_0.Analyzer(flattened_group_messages, values='')
        if result < -2 * theta or result > (domain - 1) * lambda_n + 2 * theta:
            Q[0][g] = float('-inf')
            # save_snapshot.save_q_snapshot(group_messages, "invalid_0")
        else:
            Q[0][g] = result

    # ========== (2) Higher-layer Group Detection ==========
    # level 1-logn+1
    for r in range(1, L):
        group_count = math.ceil(num_users / (lambda_n * (2 ** r)))
        Q[r] = np.zeros(group_count)
        group_size = lambda_n * (2 ** r)
        baseline_r = baseline[r]

        for g in range(group_count):
            # check validness of subgroups
            left_idx = 2 * g
            right_idx = 2 * g + 1
            if Q[r - 1][left_idx] == float('-inf') or Q[r - 1][right_idx] == float('-inf'):
                Q[r][g] = float('-inf') 
                continue

            start_idx = g * group_size
            end_idx = min((g + 1) * group_size, num_users)
            group_messages = [all_messages[i][r] for i in range(start_idx, end_idx)]
            flattened_group_messages = list(chain.from_iterable(group_messages))
            result = baseline_r.Analyzer(flattened_group_messages, values='')

            # check difference between levels
            diff = abs(result - Q[r - 1][left_idx] - Q[r - 1][right_idx])
            if r == L - 1:
                theta_2 = get_theta(baseline_r, beta_part2)
                if diff > 4 * theta + 2 * theta_2:
                    Q[r][g] = float('-inf')
                else:
                    Q[r][g] = result
            else:
                if diff > 6 * theta:
                    Q[r][g] = float('-inf')
                    # save_snapshot.save_q_snapshot(group_messages, "invalidg_r")
                else:
                    Q[r][g] = result

    # ========== (3) Recovery ==========
    A = Backtrack(max_r, Q)

    return A


def Attack():
    return [(domain - 1)] * num_users


# ------------------------------------------------------------
#  4) main: HSDP
# ------------------------------------------------------------
def HSDP(baseline, values, sorted_malicious):
    n = len(values)
    lambda_n = find_lambda(n)
    L = int(math.ceil(math.log2(n / lambda_n))) + 1

    # ========== (1) Randomization ==========

    # all_messages[i] store messages sent by user i at each level.
    all_messages = [[] for _ in range(n)]
    for i in range(n):
        if i in sorted_malicious:
            m_r = [[(domain - 1)] * n for _ in range(L)]
        else:
            m_r = LocalRandomizer(baseline, values[i])
        all_messages[i] = m_r

    # ========== (2) Analyzer: Detection + Recovery ==========
    dp_sum = Analyzer(baseline, all_messages)
    nmessages = sum(len(msg) for i, user_msgs in enumerate(all_messages)
                    if i not in sorted_malicious
                    for msg in user_msgs)
    return dp_sum, nmessages

# ------------------------------------------------------------
#  CSUZZ
# ------------------------------------------------------------
def simulateCSUZZ(values):
    dp_sums = []
    errors = []
    nmessages_per_user = []
    n = len(values[0])
    p = np.log(1 / delta) / (epsilon ** 2 * n)

    # log_term = math.log(4 / delta)
    # eps_threshold = math.sqrt(192 / n * log_term)
    #
    # if epsilon >= eps_threshold:
    #     lambda_p = (64 / (epsilon ** 2)) * log_term
    # else:
    #     lambda_p = n - (epsilon * n ** 1.5) / math.sqrt(432 * log_term)
    # p = lambda_p / n

    for t in range(times):

        # sum of attackers
        noisy_sum_attacker = 0
        for _ in sorted_malicious[t]:
            messages = Attack()
            noisy_sum_attacker += sum(messages)

        honest_user_values = [values[t][idx] for idx in range(len(values[t]))
                              if idx not in sorted_malicious[t]]

        # sum of honest users
        y_values = []
        for x in honest_user_values:
            b = np.random.binomial(1, p)
            if b == 0:
                y = x
            else:
                y = np.random.binomial(1, 0.5)
            y_values.append(y)

        noisy_sum = noisy_sum_attacker + sum(y_values)

        dp_sum = (1 / (1 - p)) * (noisy_sum - n * p / 2)

        dp_sums.append(dp_sum)
        errors.append(abs(dp_sum - sum(values[t])))
        nmessages_per_user.append(1)
    return dp_sums, errors, nmessages_per_user

# ------------------------------------------------------------
#  Initial GKMPS objects for each level
# ------------------------------------------------------------
def init_GKMPS():
    # === Initial GKMPS objects ===
    lambda_n = find_lambda(num_users)
    L = int(math.ceil(math.log2(num_users / lambda_n))) + 1
    eps_part1 = epsilon / (2 * (L - 1))  # split privacy budget
    eps_part2 = epsilon / 2
    delta_part1 = delta / (2 * (L - 1))  # split privacy budget
    delta_part2 = delta / 2
    gkmps_list = []
    for r in range(L - 1):
        # n=2^(r-1)*lambda (mind the privacy attack)
        privacy_scale = max((2 ** r - 1) / (2 ** r), 1)
        gkmps_r = GKMPS.GKMPS(n=lambda_n * (2 ** r), domain=domain - 1, epsilon=eps_part1 * privacy_scale,
                              delta=delta_part1 * privacy_scale,
                              gamma=gamma)
        gkmps_list.append(gkmps_r)
        # setting=[lambda_n * (2 ** (r - 1)),eps_part1,delta_part]
        # save_snapshot.save_1_snapshot(setting,"settinggkmps")
    gkmps_r = GKMPS.GKMPS(num_users, domain=domain - 1, epsilon=eps_part2 * (num_users - 1) / num_users,
                          delta=delta_part2 * (num_users - 1) / num_users,
                          gamma=gamma)
    gkmps_list.append(gkmps_r)
    return gkmps_list

# ------------------------------------------------------------
#  Initial BBGN objects for each level
# ------------------------------------------------------------
def init_BBGN():
    # === Initial BBGN objects ===
    lambda_n = find_lambda(num_users)
    L = int(math.ceil(math.log2(num_users / lambda_n))) + 1
    eps_part1 = epsilon / (2 * (L - 1))  # split privacy budget
    eps_part2 = epsilon / 2
    delta_part1 = delta / (2 * (L - 1))  # split privacy budget
    delta_part2 = delta / 2
    delta_part = delta / (L)
    bbgn_list = []
    for r in range(L - 1):
        privacy_scale = max((2 ** r - 1) / (2 ** r), 1)
        bbgn_r = BBGN.BBGN(n=lambda_n * (2 ** r), U=domain - 1, epsilon=eps_part1 * privacy_scale,
                           delta=delta_part1 * privacy_scale)
        bbgn_list.append(bbgn_r)
    bbgn_r = BBGN.BBGN(n=num_users, U=domain - 1, epsilon=eps_part2 * (num_users - 1) / num_users,
                       delta=delta_part2 * (num_users - 1) / num_users)
    bbgn_list.append(bbgn_r)
    # save_snapshot.save_1_snapshot(bbgn_r.m, "bbgn_m")
    return bbgn_list


# ------------------------------------------------------------
#  GKMPS protocol
# ------------------------------------------------------------
def baselineGKMPS(values):
    gkmps = GKMPS.GKMPS(n=num_users, domain=domain - 1, epsilon=epsilon, delta=delta,
                        gamma=gamma)
    gkmps_sums = []
    gkmps_errors = []
    gkmps_nmessages = []
    for i in range(times):
        messages = []
        nmessages = 0
        for j in range(num_users):
            if j in sorted_malicious[i]:
                m = Attack()
            else:
                m = gkmps.LocalRandomizer(values[i][j])
                nmessages += len(m)
            messages += m
        avg_nmessages = nmessages / (num_users - k)
        result = gkmps.Analyzer(messages, "")
        gkmps_sums.append(result)
        gkmps_errors.append(abs(result - sum(values[i])))
        gkmps_nmessages.append(avg_nmessages)
    return gkmps_sums, gkmps_errors, gkmps_nmessages


def simulateGKMPS(values):
    gkmps = GKMPS.GKMPS(n=num_users, domain=domain - 1, epsilon=epsilon, delta=delta,
                        gamma=gamma)
    gkmps_sums = []
    gkmps_errors = []
    gkmps_nmessages = []
    for t in range(times):
        # k attackers
        honest_user_values = [values[t][idx] for idx in range(len(values[t]))
                              if idx not in sorted_malicious[t]]
        honest_user_proportion = (num_users - k) / num_users
        nmessages, noisy_sum_huser = gkmps.Simulator_for_GKMPS_k(honest_user_values,
                                                                 honest_user_proportion)
        noisy_sum_attacker = 0
        for _ in sorted_malicious[t]:
            messages = Attack()
            noisy_sum_attacker += sum(messages)
        dp_sum = noisy_sum_huser + noisy_sum_attacker
        gkmps_sums.append(dp_sum)
        gkmps_errors.append(abs(dp_sum - sum(values[t])))
        gkmps_nmessages.append(nmessages / (num_users - k))
    return gkmps_sums, gkmps_errors, gkmps_nmessages


def simulateBBGN(values):
    bbgn = BBGN.BBGN(n=num_users, U=domain - 1, epsilon=epsilon, delta=delta)
    print(f"Bits per message: {math.ceil(math.log2(bbgn.domain))}")
    bbgn_sums = []
    bbgn_errors = []
    bbgn_nmessages = []
    for t in range(times):
        # k 个 attacker
        honest_user_values = [values[t][idx] for idx in range(len(values[t]))
                              if idx not in sorted_malicious[t]]
        honest_user_proportion = (num_users - k) / num_users

        # Central-DP values
        num_plus_1 = np.random.negative_binomial(honest_user_proportion, 1 - math.exp(-bbgn.epsilon / bbgn.U))
        num_minus_1 = np.random.negative_binomial(honest_user_proportion, 1 - math.exp(-bbgn.epsilon / bbgn.U))
        noisy_sum_huser = sum(honest_user_values) + num_plus_1 - num_minus_1

        noisy_sum_attacker = 0
        for _ in sorted_malicious[t]:
            messages = Attack()
            noisy_sum_attacker += sum(messages)
        dp_sum = noisy_sum_huser + noisy_sum_attacker
        bbgn_sums.append(dp_sum)
        bbgn_errors.append(abs(dp_sum - sum(values[t])))

        bbgn_nmessages.append(bbgn.m)

    return bbgn_sums, bbgn_errors, bbgn_nmessages


# ours + origin GKMPS protocol
def ours_GKMPS(values):
    gkmps_list = init_GKMPS()
    errors = []
    dp_sums = []
    nmessages_per_user = []

    for i in range(times):
        # for r in range(int(math.ceil(math.log2(num_users / lambda_n))) + 1):
        #     bath = []
        #     group_size =(lambda_n * (2 ** r))
        #     for g in range(int(num_users / group_size)):
        #         start_idx = g * group_size
        #         end_idx = min((g + 1) * group_size, num_users)
        #         bath.append(sum(values[i][start_idx:end_idx]))
        #     baths.append(bath)

        dp_sum, nmessages = HSDP(gkmps_list, values[i], sorted_malicious[i])
        dp_sums.append(dp_sum)
        errors.append(abs(dp_sum - sum(values[i])))

        nmessages_per_user.append(nmessages / (num_users - k))
        # save_snapshot.save_q_snapshot(baths, "bath")
    return dp_sums, errors, nmessages_per_user


def ours_BBGN(values):
    bbgn_list = init_BBGN()
    errors = []
    dp_sums = []
    nmessages_per_user = []
    for i in range(times):
        dp_sum, nmessages = HSDP(bbgn_list, values[i], sorted_malicious[i])
        dp_sums.append(dp_sum)
        errors.append(abs(dp_sum - sum(values[i])))

        nmessages_per_user.append(nmessages / (num_users - k))
    return dp_sums, errors, nmessages_per_user


def baselineBBGN(values):
    bbgn = BBGN.BBGN(n=num_users, U=domain - 1, epsilon=epsilon, delta=delta)
    # save_snapshot.save_1_snapshot(bbgn.m, "bbgn_m")
    # setting = [num_users, epsilon, delta, domain - 1, gamma]
    # save_snapshot.save_1_snapshot(setting,"settinggkmps")
    bbgn_sums = []
    bbgn_errors = []
    bbgn_nmessages = []
    for i in range(times):
        messages = []
        nmessages = 0
        for j in range(num_users):
            if j in sorted_malicious[i]:
                m = Attack()
            else:
                m = bbgn.LocalRandomizer(values[i][j])
                nmessages += len(m)
            messages += m
        avg_nmessages = nmessages / (num_users - k)
        result = bbgn.Analyzer(messages, "")
        bbgn_sums.append(result)
        bbgn_errors.append(abs(result - sum(values[i])))
        bbgn_nmessages.append(avg_nmessages)
    return bbgn_sums, bbgn_errors, bbgn_nmessages


def processResult(dp_sums, real_sums, errors, beta):
    times = len(dp_sums)
    cut_count = int(beta * times / 2)
    combined = list(zip(errors, dp_sums, real_sums))
    combined.sort()
    start = cut_count
    end = len(combined) - cut_count
    trimmed = combined[start:end]
    if trimmed:
        errors, dp_sums, real_sums = zip(*trimmed)
        errors = list(errors)
        dp_sums = list(dp_sums)
        real_sums = list(real_sums)
    else:
        errors = []
        dp_sums = []
        real_sums = []
    return dp_sums, real_sums, errors


# ========================== Simulator =========================
def get_theta_sim(baseline_name, eps, domain, beta):
    if baseline_name == "BBGN":
        return ((domain) / eps) * math.log((2 * math.exp(eps)) / (beta * (math.exp(eps) + 1)))
    elif baseline_name == "GKMPS":
        return ((domain) / ((1 - gamma) * eps)) * math.log(
            (2 * math.exp(((1 - gamma) * eps))) / (beta * (math.exp(((1 - gamma) * eps)) + 1)))
    else:
        return 0.0


# ----------------------------------------------------
# Simulated Analyzer: Detect Q[[]] + Backtrack
# baseline_name: distinct for get_theta_sim
# Q: [L][group_count], store result for each group in the binary tree
# output: result of the root node after recovery
# ----------------------------------------------------
def simulate_analyzer(baseline_name, Q, L, lambda_n, beta, domain, eps_part1, eps_part2):
    if k == 0:
        return Q[L - 1][0]

    beta_part1 = beta / 2 / (2 ** L - 2)
    beta_part2 = beta / 2
    theta1 = get_theta_sim(baseline_name, eps_part1, domain - 1, beta_part1)
    theta2 = get_theta_sim(baseline_name, eps_part2, domain - 1, beta_part2)

    # ========== (1) Detection (r=0) ==========
    group_count_0 = len(Q[0])
    for g in range(group_count_0):
        val = Q[0][g]
        if val < -2 * theta1 or val > (domain - 1) * lambda_n + 2 * theta1:
            Q[0][g] = float('-inf')

    # ========== (2) Detection (r>0) ==========
    for r in range(1, L):
        group_count_r = len(Q[r])
        for g in range(group_count_r):
            left_idx = 2 * g
            right_idx = 2 * g + 1
            if Q[r - 1][left_idx] == float('-inf') or Q[r - 1][right_idx] == float('-inf'):
                Q[r][g] = float('-inf')
                continue
            diff = abs(Q[r][g] - (Q[r - 1][left_idx] + Q[r - 1][right_idx]))

            if r == L - 1:  # 顶层
                if diff > 4 * theta1 + 2 * theta2:
                    Q[r][g] = float('-inf')
            else:
                if diff > 6 * theta1:
                    Q[r][g] = float('-inf')

    # ========== (3) Backtrack 恢复 ==========
    dp_sum = Backtrack(L - 1, Q)
    return dp_sum


# ----------------------------------------------------
# simulated version of “ours + BBGN”
# ----------------------------------------------------
def simulate_ours_BBGN(values):
    n = len(values[0])  # values: [times][n]

    lambda_n = find_lambda(n)
    L = int(math.ceil(math.log2(n / lambda_n))) + 1
    eps_part1 = epsilon / 2 / (L - 1)  # split privacy budget for layers except the top
    eps_part2 = epsilon / 2

    dp_sums = []
    errors = []
    nmessages_per_user = []

    for t in range(times):

        # ========== (1) declare Q[r] ==========
        Q = []
        for r in range(L):
            group_size = lambda_n * (2 ** r)
            group_count = int(math.ceil(n / group_size))
            Q.append(np.zeros(group_count, dtype=float))

        # ========== (2) construct Q[r] ==========
        #  real sum + noise of BBGN

        total_messages = 0
        for r in range(L):
            group_size = lambda_n * (2 ** r)
            group_count = len(Q[r])
            eps_r = eps_part2 if (r == L - 1) else eps_part1
            eps_r = eps_r * max((2 ** r - 1) / (2 ** r), 1)

            for g in range(group_count):
                start_index = g * group_size
                end_index = min((g + 1) * group_size, n)

                # computer the #attackers in the current group
                left = bisect.bisect_left(sorted_malicious[t], start_index)
                right = bisect.bisect_right(sorted_malicious[t], end_index)
                attackers_count_g = right - left

                # compute the results sum of poisoning attackers
                noisy_sum_attacker = 0
                real_sum_attacker = 0
                for i in sorted_malicious[t][left:right]:
                    messages = Attack()
                    noisy_sum_attacker += sum(messages)
                    real_sum_attacker += values[t][i]

                # real sum
                real_sum_huser = sum(values[t][start_index:end_index]) - real_sum_attacker

                plus_noise = np.random.negative_binomial(
                    (group_size - attackers_count_g) / group_size, 1 - math.exp(-eps_r / max(1, domain - 1))
                )
                minus_noise = np.random.negative_binomial(
                    (group_size - attackers_count_g) / group_size, 1 - math.exp(-eps_r / max(1, domain - 1))
                )
                noisy_sum_huser = real_sum_huser + plus_noise - minus_noise
                noisy_sum = noisy_sum_huser + noisy_sum_attacker
                Q[r][g] = noisy_sum

            # messages
            total_messages += max(3, int(math.ceil((2 * sigma + math.log2(group_size * (domain - 1) * 10)) / (
                    math.log2(group_size) - math.log2(math.e)) + 1)))
            if r == L - 1:
                print(f"U:{domain}, n:{n}")
                print(math.ceil(math.log2(10 * (domain - 1) * n)))
                print(f"n/lambda={n / lambda_n},{math.ceil(math.log2(2 * n / lambda_n - 1))}")
                print(
                    f"Bits per message: {math.ceil(math.log2(10 * (domain - 1) * n) + math.ceil(math.log2(2 * n / lambda_n - 1)))}")
        nmessages_per_user.append(total_messages)

        # ========== (3) simulate_analyzer: detection + recovery ==========
        dp_sum = simulate_analyzer(
            baseline_name="BBGN",
            Q=Q,
            L=L,
            lambda_n=lambda_n,
            beta=beta,
            domain=domain,
            eps_part1=eps_part1,
            eps_part2=eps_part2
        )

        # ========== (4) compute error ==========
        true_sum = sum(values[t])
        err = abs(dp_sum - true_sum)
        errors.append(err)
        dp_sums.append(dp_sum)

    return dp_sums, errors, nmessages_per_user


def simulate_ours_GKMPS(values):
    n = len(values[0])
    lambda_n = find_lambda(n)
    L = int(math.ceil(math.log2(n / lambda_n))) + 1

    gkmps_list = init_GKMPS()

    dp_sums, errors, nmessages_per_user = [], [], []

    for t in range(times):

        Q = []
        for r in range(L):
            group_size = lambda_n * (2 ** r)
            group_count = int(math.ceil(n / group_size))
            Q.append(np.zeros(group_count, dtype=float))

        total_messages_all_users = 0
        for r in range(L):
            group_size = lambda_n * (2 ** r)
            group_count = len(Q[r])
            test = 0
            for g in range(group_count):
                start_index = g * group_size
                end_index = min((g + 1) * group_size, n)

                left = bisect.bisect_left(sorted_malicious[t], start_index)
                right = bisect.bisect_right(sorted_malicious[t], end_index)
                attackers_count_g = right - left
                honest_user_proportion = (group_size - attackers_count_g) / group_size

                noisy_sum_attacker = 0
                for _ in sorted_malicious[t][left:right]:
                    messages = Attack()
                    noisy_sum_attacker += sum(messages)

                honest_user_values = [values[t][idx] for idx in range(start_index, end_index)
                                      if idx not in sorted_malicious[t]]
                nmessages_group, noisy_sum_huser = gkmps_list[r].Simulator_for_GKMPS_k(honest_user_values,
                                                                                       honest_user_proportion)
                # if r == L - 1:
                #     print(gkmps_list[r].epsilon)
                Q[r][g] = noisy_sum_huser + noisy_sum_attacker
                total_messages_all_users += nmessages_group
                test += nmessages_group
            # print(f"group size:{group_size}, layer:{r}, messages: {test / n}")
        nmessages_per_user.append(total_messages_all_users / (n - k))

        dp_sum = simulate_analyzer(
            baseline_name="GKMPS",
            Q=Q,
            L=L,
            lambda_n=lambda_n,
            beta=beta,
            domain=domain,
            eps_part1=epsilon / 2 / (L - 1),
            eps_part2=epsilon / 2
        )
        # ========== (4) compute error ==========
        true_sum = sum(values[t])
        err = abs(dp_sum - true_sum)
        errors.append(err)
        dp_sums.append(dp_sum)

    return dp_sums, errors, nmessages_per_user
