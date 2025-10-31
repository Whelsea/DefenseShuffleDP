# --------------------------------------
# Levels:
#   r = 1 → User-level (each user separately)
#   r = 2 → Block-level (√n per block)
#   r = 3 → Output-level (all users together)
# --------------------------------------

import math
from itertools import chain
import advanced_HSDP  
import numpy as np
import bisect

def num_users():
    return advanced_HSDP.num_users

def domain():
    return advanced_HSDP.domain

def epsilon():
    return advanced_HSDP.epsilon

def delta():
    return advanced_HSDP.delta

def beta():
    return advanced_HSDP.beta

def k():
    return advanced_HSDP.k

def times():
    return advanced_HSDP.times

def sorted_malicious():
    return advanced_HSDP.sorted_malicious

def gamma():
    return advanced_HSDP.gamma

def sigma():
    return advanced_HSDP.sigma


# baseline imports
import BBGN
import GKMPS


# ------------------------------------------------------------
# Initialize BBGN baseline objects
# ------------------------------------------------------------
def init_BBGN():
    n = num_users()
    d = domain() - 1

    eps_user = epsilon() / 3
    eps_block = (epsilon() / 3) * ((math.sqrt(n) - 1) / math.sqrt(n))
    eps_output = (epsilon() / 3) * ((n - 1) / n)

    delta_user = delta() / 3
    delta_block = (delta() / 3) * ((math.sqrt(n) - 1) / math.sqrt(n))
    delta_output = (delta() / 3) * ((n - 1) / n)

    # user-level
    bbgn_user = BBGN.BBGN(n=1, U=d, epsilon=eps_user, delta=delta_user)
    # block-level
    bbgn_block = BBGN.BBGN(n=int(math.sqrt(n)), U=d, epsilon=eps_block, delta=delta_block)
    # output-level
    bbgn_output = BBGN.BBGN(n=n, U=d, epsilon=eps_output, delta=delta_output)

    return [bbgn_user, bbgn_block, bbgn_output]


# ------------------------------------------------------------
# Initialize GKMPS baseline objects
# ------------------------------------------------------------
def init_GKMPS():
    n = num_users()
    d = domain() - 1

    eps_user = epsilon()/ 3
    eps_block = (epsilon() / 3) * ((math.sqrt(n) - 1) / math.sqrt(n))
    eps_output = (epsilon() / 3) * ((n - 1) / n)

    delta_user = delta() / 3
    delta_block = (delta() / 3) * ((math.sqrt(n) - 1) / math.sqrt(n))
    delta_output = (delta() / 3) * ((n - 1) / n)

    gkmps_user = GKMPS.GKMPS(n=1, domain=d, epsilon=eps_user, delta=delta_user, gamma=advanced_HSDP.gamma)
    gkmps_block = GKMPS.GKMPS(n=int(math.sqrt(n)), domain=d, epsilon=eps_block, delta=delta_block, gamma=advanced_HSDP.gamma)
    gkmps_output = GKMPS.GKMPS(n=n, domain=d, epsilon=eps_output, delta=delta_output, gamma=advanced_HSDP.gamma)

    return [gkmps_user, gkmps_block, gkmps_output]


# ------------------------------------------------------------
# Compute error threshold
# ------------------------------------------------------------
def get_theta(baseline, beta_val):
    # BBGN
    if baseline.name == "BBGN":
        return (baseline.U / baseline.epsilon) * math.log(
            (2 * math.exp(baseline.epsilon)) / (beta_val * (math.exp(baseline.epsilon) + 1)))
    # GKMPS
    if baseline.name == "GKMPS":
        return (baseline.domain / baseline.epsilonstar) * math.log(
            (2 * math.exp(baseline.epsilonstar)) / (beta_val * (math.exp(baseline.epsilonstar) + 1)))


# ------------------------------------------------------------
# Local Randomizer: Given a baseline, return randomized messages
#                   for the user at three hierarchical levels
# ------------------------------------------------------------
def LocalRandomizer(baselines, value):
    messages_per_user = []
    for b in baselines:
        m = b.LocalRandomizer(value)
        print(value)
        print(m)
        messages_per_user.append(m)
    return messages_per_user  # [msg_user, msg_block, msg_output]


# ------------------------------------------------------------
# Analyzer of BSDP
# ------------------------------------------------------------
def Analyzer(baselines, all_messages):
    n = num_users()
    bsize = int(math.isqrt(n))  # block size = √n
    n_blocks = n // bsize

    # privacy budgets for error calculation
    beta_user_block = beta() / (2 * (bsize + n))  # shared by user/block
    beta_output = beta() / 2

    theta_user = get_theta(baselines[0], beta_user_block)
    theta_block = get_theta(baselines[1], beta_user_block)
    theta_output = get_theta(baselines[2], beta_output)
    print("theta:", theta_user)
    print("theta:", theta_block)
    print("theta:", theta_output)


    # ---------- 1. User-Level Detection ----------
    Q_user = [0.0] * n
    for i in range(n):
        result = baselines[0].Analyzer(all_messages[i][0], values='')
        if result < -theta_user or result > (domain() - 1) + theta_user:
            Q_user[i] = float('-inf')
        else:
            Q_user[i] = result

    # ---------- 2. Block-Level Detection ----------
    Q_block = [0.0] * n_blocks
    for b in range(n_blocks):
        # block messages
        start_idx = b * bsize
        end_idx = (b + 1) * bsize
        # group_messages = [all_messages[i][1] for i in range(start_idx, end_idx)]
        # print(len(group_messages))
        #
        # flat_msgs = list(chain.from_iterable(group_messages))
        #
        # result = baselines[1].Analyzer(flat_msgs, values='')
        # collect block-layer messages for users in this block
        group_messages = [all_messages[i][1] for i in range(start_idx, end_idx)]
        flattened_group_messages = list(chain.from_iterable(group_messages))

        # protocol result (baseline's Analyzer)
        result = baselines[1].Analyzer(flattened_group_messages, values='')
        # result= sum(flat_msgs)
        # print(len(flat_msgs))
        # print(sum(flat_msgs))
        # print(flat_msgs[10:25])
        print(f"block {b} result:{result}")

        # check children
        if any(Q_user[i] == float('-inf') for i in range(start_idx, end_idx)):
            Q_block[b] = float('-inf')
        else:
            diff = abs(result - sum(Q_user[start_idx:end_idx]))
            if diff > bsize * theta_user + theta_block:
                Q_block[b] = float('-inf')
            else:
                Q_block[b] = result
        if Q_block[b] == float('-inf'):
            print(f"block {b} is detected")

    # ---------- 3. Output-Level Detection ----------
    all_output_msgs = [all_messages[i][2] for i in range(n)]
    flat_output_msgs = list(chain.from_iterable(all_output_msgs))
    result = baselines[2].Analyzer(flat_output_msgs, values='')

    if any(qb == float('-inf') for qb in Q_block) or \
       abs(result - sum(qb for qb in Q_block if qb != float('-inf'))) > math.sqrt(n) * theta_block + theta_output:
        Q_output = float('-inf')
    else:
        Q_output = result

    # ---------- 4. Recovery ----------
    for i in range(n):
        if Q_user[i] == float('-inf'):
            Q_user[i] = 0

    for b in range(n_blocks):
        if Q_block[b] == float('-inf'):
            start_idx = b * bsize
            end_idx = (b + 1) * bsize
            Q_block[b] = sum(Q_user[start_idx:end_idx])

    if Q_output == float('-inf'):
        Q_output = sum(Q_block)

    return Q_output


# ------------------------------------------------------------
# BSDP
# ------------------------------------------------------------
def BSDP(baselines, values, malicious):
    n = num_users()
    all_messages = []
    nmessages = 0

    for i in range(n):
        if i in malicious:
            # The corrupted user sends messages with U values at all three layers.
            print(f"yes there is an atta {n}")
            m_user = [(domain() - 1)] * n
            m_block = [(domain() - 1)] * n * n
            m_output = [(domain() - 1)] * n
            print(len(m_user))
            all_messages.append([m_user, m_block, m_output])
        else:
            m = LocalRandomizer(baselines, values[i])
            all_messages.append(m)
            nmessages += sum(len(level) for level in m)

    dp_sum = Analyzer(baselines, all_messages)
    return dp_sum, nmessages


# ------------------------------------------------------------
# BSDP+BBGN
# ------------------------------------------------------------
def BSDP_BBGN(values):
    baselines = init_BBGN()
    errors = []
    dp_sums = []
    nmessages_per_user = []
    sm = sorted_malicious()

    for t in range(times()):
        dp_sum, nmessages = BSDP(baselines, values[t], sm[t])
        dp_sums.append(dp_sum)
        errors.append(abs(dp_sum - sum(values[t])))
        nmessages_per_user.append(nmessages / (num_users() - k()))

    return dp_sums, errors, nmessages_per_user


# ------------------------------------------------------------
# BSDP+GKMPS
# ------------------------------------------------------------
def BSDP_GKMPS(values):
    baselines = init_GKMPS()
    errors = []
    dp_sums = []
    nmessages_per_user = []
    sm = sorted_malicious()

    for t in range(times()):
        dp_sum, nmessages = BSDP(baselines, values[t], sm[t])
        dp_sums.append(dp_sum)
        errors.append(abs(dp_sum - sum(values[t])))
        nmessages_per_user.append(nmessages / (num_users() - k()))

    return dp_sums, errors, nmessages_per_user

def simulate_BSDP_BBGN(values):
    baselines = init_BBGN()
    errors = []
    dp_sums = []
    nmessages_per_user = []
    attackernum=k()
    sm = sorted_malicious()
    rounds=times()
    n = num_users()
    bsize = int(math.isqrt(n))  # block size = √n
    n_blocks = n // bsize

    beta_user_block = beta() / (2 * (bsize + n))  # shared by user/block
    beta_output = beta() / 2

    theta_user = get_theta(baselines[0], beta_user_block)
    theta_block = get_theta(baselines[1], beta_user_block)
    theta_output = get_theta(baselines[2], beta_output)



    for t in range(rounds):
        Q_user = [0.0] * n
        nmessages = 0
        bbgn_user = baselines[0]
        nmessages += bbgn_user.m
        bbgn_block = baselines[1]
        nmessages += bbgn_block.m
        bbgn_output = baselines[2]
        nmessages += bbgn_output.m

        for u in range(n):
            if u not in sm[t]:
                # Central-DP values
                num_plus_1 = np.random.negative_binomial(1.0, 1 - math.exp(-bbgn_user.epsilon / bbgn_user.U))
                num_minus_1 = np.random.negative_binomial(1.0, 1 - math.exp(-bbgn_user.epsilon / bbgn_user.U))
                Q_user[u] = values[t][u] + num_plus_1 - num_minus_1
            else:
                Q_user[u] = n

            if Q_user[u] < -theta_user or Q_user[u] > (domain() - 1) + theta_user:
                Q_user[u] = 0



        Q_block = [0.0] * n_blocks
        for b in range(n_blocks):
            start_index = b * bsize
            end_index = min((b + 1) * bsize, n)

            # Check whether this block contains an attacker
            left = bisect.bisect_left(sm[t], start_index)
            right = bisect.bisect_right(sm[t], end_index - 1)
            has_attacker = (right - left) > 0

            if not has_attacker:
                # Honest block: add normal (DP) noise
                num_plus_1 = np.random.negative_binomial(1, 1 - math.exp(-bbgn_block.epsilon / bbgn_block.U))
                num_minus_1 = np.random.negative_binomial(1, 1 - math.exp(-bbgn_block.epsilon / bbgn_block.U))
                Q_block[b] = sum(values[t][start_index:end_index]) + num_plus_1 - num_minus_1

                # If the deviation is too large, revert to aggregated user-level values
                diff = abs(Q_block[b] - sum(Q_user[start_index:end_index]))
                if diff > bsize * theta_user + theta_block:
                    Q_block[b] = sum(Q_user[start_index:end_index])
            else:
                # Malicious block: directly use the sum of user-level results
                Q_block[b] = sum(Q_user[start_index:end_index])




        if attackernum > 0:
            Q_output = sum(Q_block)
        else:
            num_plus_1 = np.random.negative_binomial(1, 1 - math.exp(-bbgn_output.epsilon / bbgn_output.U))
            num_minus_1 = np.random.negative_binomial(1, 1 - math.exp(-bbgn_output.epsilon / bbgn_output.U))
            Q_output = sum(values[t])+ num_plus_1 - num_minus_1

        dp_sums.append(Q_output)
        nmessages_per_user.append(nmessages)
        errors.append(abs(Q_output - sum(values[t])))

    return dp_sums, errors, nmessages_per_user

def simulate_BSDP_BBGN_speed(values):
    baselines = init_BBGN()
    bbgn_user, bbgn_block, bbgn_output = baselines
    n = num_users()
    U = domain()
    BETA = beta()
    ATTACKERNUM = k()
    ROUNDS = times()
    SM = [np.array(x, dtype=int) for x in sorted_malicious()]

    bsize = int(math.isqrt(n))
    n_blocks = n // bsize

    beta_user_block = BETA / (2 * (bsize + n))
    beta_output = BETA / 2

    theta_user = get_theta(bbgn_user, beta_user_block)
    theta_block = get_theta(bbgn_block, beta_user_block)
    theta_output = get_theta(bbgn_output, beta_output)

    errors = np.zeros(ROUNDS)
    dp_sums = np.zeros(ROUNDS)
    nmessages_per_user = np.zeros(ROUNDS)

    for t in range(ROUNDS):
        mask_honest = np.ones(n, dtype=bool)
        mask_honest[SM[t]] = False

        # user-level noise
        num_plus = np.random.negative_binomial(1.0, 1 - np.exp(-bbgn_user.epsilon / bbgn_user.U), size=n)
        num_minus = np.random.negative_binomial(1.0, 1 - np.exp(-bbgn_user.epsilon / bbgn_user.U), size=n)
        Q_user = np.where(mask_honest, values[t] + num_plus - num_minus, n)
        Q_user[(Q_user < -theta_user) | (Q_user > (U - 1) + theta_user)] = 0

        # block-level aggregation
        Q_block = np.zeros(n_blocks)
        cum_values = np.cumsum(values[t])
        cum_Q_user = np.cumsum(Q_user)

        for b in range(n_blocks):
            start, end = b * bsize, min((b + 1) * bsize, n)
            has_attacker = np.any((SM[t] >= start) & (SM[t] < end))
            block_sum_val = cum_values[end-1] - (cum_values[start-1] if start > 0 else 0)
            block_sum_user = cum_Q_user[end-1] - (cum_Q_user[start-1] if start > 0 else 0)

            if not has_attacker:
                num_plus = np.random.negative_binomial(1, 1 - np.exp(-bbgn_block.epsilon / bbgn_block.U))
                num_minus = np.random.negative_binomial(1, 1 - np.exp(-bbgn_block.epsilon / bbgn_block.U))
                Q_block[b] = block_sum_val + num_plus - num_minus
                if abs(Q_block[b] - block_sum_user) > bsize * theta_user + theta_block:
                    Q_block[b] = block_sum_user
            else:
                Q_block[b] = block_sum_user

        # output-level
        if ATTACKERNUM > 0:
            Q_output = np.sum(Q_block)
        else:
            num_plus = np.random.negative_binomial(1, 1 - np.exp(-bbgn_output.epsilon / bbgn_output.U))
            num_minus = np.random.negative_binomial(1, 1 - np.exp(-bbgn_output.epsilon / bbgn_output.U))
            Q_output = np.sum(values[t]) + num_plus - num_minus

        dp_sums[t] = Q_output
        nmessages_per_user[t] = bbgn_user.m + bbgn_block.m + bbgn_output.m
        errors[t] = abs(Q_output - np.sum(values[t]))

    return dp_sums, errors, nmessages_per_user


from types import SimpleNamespace

def simulate_BSDP_GKMPS_speed(values):
    """
    Simulated BSDP + GKMPS:
    - Preserves the 3-layer BSDP hierarchy (user / block / output)
    - User & block layers add central noise based on epsilonstar / U
    - Output layer adds zero-sum noise (significantly increases message count)
    - Message statistics are computed via GKMPS.EstimateMessageNumber()
    """

    # ===== Initialize three-layer GKMPS baselines =====
    baselines = init_GKMPS()
    gkmps_user, gkmps_block, gkmps_output = baselines
    gkmps = gkmps_user  # Used to access epsilonstar, epsilon1, epsilon2, delta1, delta2, t, etc.

    # ===== Fixed parameters =====
    n       = num_users()
    U       = gkmps.U
    ROUNDS  = times()
    ATTACKN = k()
    SM      = [np.array(x, dtype=int) for x in sorted_malicious()]

    bsize    = int(math.isqrt(n))
    n_blocks = n // bsize

    BETA = beta()
    beta_user_block = BETA / (2 * (bsize + n))
    beta_output     = BETA / 2

    theta_user   = get_theta(gkmps_user,   beta_user_block)
    theta_block  = get_theta(gkmps_block,  beta_user_block)
    theta_output = get_theta(gkmps_output, beta_output)

    errors              = np.zeros(ROUNDS)
    dp_sums             = np.zeros(ROUNDS)
    nmessages_per_user  = np.zeros(ROUNDS)

    # ===== Parameters for the negative binomial distribution =====
    p_star = 1 - math.exp(-gkmps_user.epsilonstar / U)

    # ===== Main simulation loop =====
    for t in range(ROUNDS):
        nmessages = 0

        # ---------- User layer ----------
        mask_honest = np.ones(n, dtype=bool)
        mask_honest[SM[t]] = False

        num_plus  = np.random.negative_binomial(1.0, p_star, size=n)
        num_minus = np.random.negative_binomial(1.0, p_star, size=n)

        Q_user = np.where(mask_honest, values[t] + num_plus - num_minus, n)
        Q_user[(Q_user < -theta_user) | (Q_user > (U - 1) + theta_user)] = 0

        # ---------- Block layer ----------
        Q_block = np.zeros(n_blocks)
        cum_values = np.cumsum(values[t])
        cum_Q_user = np.cumsum(Q_user)

        for b in range(n_blocks):
            start, end = b * bsize, min((b + 1) * bsize, n)
            has_attacker = np.any((SM[t] >= start) & (SM[t] < end))

            block_sum_val  = cum_values[end - 1] - (cum_values[start - 1] if start > 0 else 0)
            block_sum_user = cum_Q_user[end - 1] - (cum_Q_user[start - 1] if start > 0 else 0)

            if not has_attacker:
                # Honest block: add central noise
                num_plus_b  = np.random.negative_binomial(1, p_star)
                num_minus_b = np.random.negative_binomial(1, p_star)
                Q_block[b] = block_sum_val + num_plus_b - num_minus_b

                # Validate against the user-level aggregation consistency check
                if abs(Q_block[b] - block_sum_user) > bsize * theta_user + theta_block:
                    Q_block[b] = block_sum_user
            else:
                # Malicious block: directly use the user-level aggregation
                Q_block[b] = block_sum_user

        # ---------- Output layer ----------
        if ATTACKN > 0:
            # If attackers exist, aggregate block-level outputs
            Q_output = float(np.sum(Q_block))
        else:
            # Add zero-sum central noise for the final output
            num_plus_o  = np.random.negative_binomial(1, p_star)
            num_minus_o = np.random.negative_binomial(1, p_star)
            Q_output = float(np.sum(values[t]) + num_plus_o - num_minus_o)

        # ---------- Result statistics ----------
        dp_sums[t] = Q_output
        errors[t] = abs(Q_output - float(np.sum(values[t])))

        # Compute the number of messages per user using GKMPS.EstimateMessageNumber()
        base_msgs_user   = gkmps_user.EstimateMessageNumber()
        base_msgs_block  = gkmps_block.EstimateMessageNumber()
        base_msgs_output = gkmps_output.EstimateMessageNumber()
        base_msgs_per_user = base_msgs_user + base_msgs_block + base_msgs_output
        nmessages_per_user[t] = base_msgs_per_user

    return dp_sums, errors, nmessages_per_user
