import math
import numpy as np
from itertools import chain
import BBGN
import GKMPS
import advanced_HSDP

values=[]
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
#
# def values():
#     return advanced_HSDP.values


# ------------------------------------------------------------
#  Initial GKMPS objects for each level
# ------------------------------------------------------------
def init_GKMPS():
    gkmps = GKMPS.GKMPS(n=1, domain=domain() - 1, epsilon=epsilon(),
                              delta=delta(),
                              gamma=gamma())
    return gkmps

# ------------------------------------------------------------
#  Initial BBGN objects for each level
# ------------------------------------------------------------
def init_BBGN():
    bbgn = BBGN.BBGN(n=1, U=domain() - 1, epsilon=epsilon(),
                           delta=delta())
    return bbgn




def get_theta(baseline, beta):
    if baseline.name == "GKMPS":
        theta = (baseline.domain / baseline.epsilonstar) * math.log(
            (2 * math.exp(baseline.epsilonstar)) / (beta * (math.exp(baseline.epsilonstar) + 1)))
    if baseline.name == "BBGN":
        theta = (baseline.U / baseline.epsilon) * math.log(
            (2 * math.exp(baseline.epsilon)) / (beta * (math.exp(baseline.epsilon) + 1)))
    return theta



def LocalRandomizer(baseline, value):
    messages = []
    m = baseline.LocalRandomizer(value)
    # messages.append(m)
    return m

# ------------------------------------------------------------
#  3) Analyzer of HSDP
#   Input:
#       - all_messages[i]: 2D array [i][]
#   Output: A (final aggregated result)
# ------------------------------------------------------------
def Analyzer(baseline, all_messages):
    # ========== (1) Bottom-layer Detection (r = 0) ==========

    Q = [0.0] * num_users()
    beta_part = beta() / num_users()
    theta = 100 * get_theta(baseline, beta_part)
    # print("theta:", theta)
    A = 0

    # Track per-user intermediate error (for debugging or analysis)
    per_user_err = []

    for i in range(num_users()):
        # Compute the result for user i (group g)
        result = baseline.Analyzer(all_messages[i], values='')
        if result < -theta or result > (domain() - 1) + theta:
            Q[i] = float('-inf')
            # Optionally record invalid users here (e.g., save_q_snapshot)
        else:
            A += result
          
    return A


# ------------------------------------------------------------
#  4) SUSDP Protocol
#   Input:
#       - baseline: baseline mechanism instance
#       - values: list of true user values
#       - sorted_malicious: indices of malicious users
#   Output:
#       - dp_sum: final aggregated noisy sum
#       - nmessages: number of messages sent by honest users
# ------------------------------------------------------------
def SUSDP(baseline, values, sorted_malicious):
    # all_messages[i] stores the messages sent by user i at all levels
    all_messages = []
    nmessages = 0  # count messages from honest users only

    for i in range(num_users()):
        if i in sorted_malicious:
            # Malicious users send n fake messages, all with value U
            m = [(domain() - 1)] * num_users()
        else:
            # Honest users send randomized messages via LocalRandomizer
            m = LocalRandomizer(baseline, values[i])
            nmessages += len(m)
        all_messages.append(m)

    dp_sum = Analyzer(baseline, all_messages)
    return dp_sum, nmessages


def SUSDP_BBGN(values):
    bbgn = init_BBGN()
    errors = []
    dp_sums = []
    nmessages_per_user = []
    for i in range(times()):
        dp_sum, nmessages = SUSDP(bbgn, values[i], sorted_malicious()[i])
        dp_sums.append(dp_sum)
        errors.append(abs(dp_sum - sum(values[i])))
        nmessages_per_user.append(nmessages / (num_users() - k()))
    print(dp_sums)
    return dp_sums, errors, nmessages_per_user


def Simulate_SUSDP_BBGN(values):
    bbgn = init_BBGN()
    errors = []
    dp_sums = []
    nmessages_per_user = []
    for i in range(times()):
        dp_sum = 0
        nmessages =0
        for u in range(num_users()):
            if u not in sorted_malicious()[i]:
                nmessages += bbgn.n * bbgn.m
                # Central-DP values
                num_plus_1 = np.random.negative_binomial(1.0, 1 - math.exp(-bbgn.epsilon / bbgn.U))
                num_minus_1 = np.random.negative_binomial(1.0, 1 - math.exp(-bbgn.epsilon / bbgn.U))
                dp_sum = dp_sum + values[i][u] + num_plus_1 - num_minus_1
        dp_sums.append(dp_sum)
        errors.append(abs(dp_sum - sum(values[i])))
        nmessages_per_user.append(nmessages / (num_users() - k()))
    return dp_sums, errors, nmessages_per_user

def Simulate_SUSDP_BBGN_speed(values):
    bbgn = init_BBGN()
    errors, dp_sums, nmessages_per_user = [], [], []

    # ======= Pre-computed fixed parameters =======
    n = num_users()
    U = bbgn.U
    eps = bbgn.epsilon
    m_per_user = bbgn.n * bbgn.m
    p = 1 - math.exp(-eps / U)   # Negative binomial distribution parameter
    mal_lists = sorted_malicious()

    for t in range(times()):
        mal_set = set(mal_lists[t])
        honest_num = n - len(mal_set)
        honest_values = np.array(values[t])[list(set(range(n)) - mal_set)]

        # ======= Vectorized noise generation =======
        num_plus_1 = np.random.negative_binomial(1.0, p, size=honest_num)
        num_minus_1 = np.random.negative_binomial(1.0, p, size=honest_num)

        # ======= Vectorized noisy aggregation =======
        noisy_vals = honest_values + num_plus_1 - num_minus_1
        dp_sum = np.sum(noisy_vals)

        # ======= Message count estimation =======
        nmessages = honest_num * m_per_user

        # ======= Record results =======
        dp_sums.append(dp_sum)
        errors.append(abs(dp_sum - np.sum(values[t])))
        nmessages_per_user.append(nmessages / (n - k()))

    return dp_sums, errors, nmessages_per_user



def SUSDP_GKMPS(values):
    gkmps = init_GKMPS()
    errors = []
    dp_sums = []
    nmessages_per_user = []
    for i in range(times()):
        dp_sum, nmessages = SUSDP(gkmps, values[i], sorted_malicious()[i])
        dp_sums.append(dp_sum)
        errors.append(abs(dp_sum - sum(values[i])))
        nmessages_per_user.append(nmessages / (num_users() - k()))
    return dp_sums, errors, nmessages_per_user


def Simulate_SUSDP_GKMPS(values):
    gkmps = init_GKMPS()
    errors = []
    dp_sums = []
    nmessages_per_user = []
    for i in range(times()):
        dp_sum = 0
        nmessages =0
        for u in range(num_users()):
            if u not in sorted_malicious()[i]:
                rvalue = gkmps.RandomizedRounding(values[i][u])
                if (rvalue != 0):
                    nmessages += 1
                dp_sum += rvalue

                # Central-DP values
                num_plus_1 = np.random.negative_binomial(1.0, 1 - math.exp(-gkmps.epsilonstar / gkmps.U))
                num_minus_1 = np.random.negative_binomial(1.0, 1 - math.exp(-gkmps.epsilonstar / gkmps.U))
                nmessages += num_plus_1 + num_minus_1
                dp_sum = dp_sum + num_plus_1 - num_minus_1

                # Zero-sum DP values
                z = np.zeros(2 * gkmps.U + 1)
                for i in range(-gkmps.U, gkmps.U + 1, 1):
                    if i == 0 or i == -1:
                        continue
                    if i == 1:
                        z[i] += np.random.negative_binomial(3 * (1 + np.log(1 / gkmps.delta1)),
                                                            1 - math.exp(-0.2 * gkmps.epsilon1 / gkmps.U))
                    z[i] += np.random.negative_binomial(3 * (1 + np.log((2 * gkmps.U - 1) / gkmps.delta2)),
                                                        1 - math.exp(-0.1 * gkmps.epsilon2 / gkmps.t[i]))

                for i in range(-gkmps.U, gkmps.U + 1, 1):
                    if i == 0 or i == -1:
                        continue
                    elif i == 1:
                        nmessages += 2 * int(z[i])
                    else:
                        nmessages += 3 * int(z[i])
        dp_sums.append(dp_sum)
        errors.append(abs(dp_sum - sum(values[i])))
        nmessages_per_user.append(nmessages / (num_users() - k()))
    return dp_sums, errors, nmessages_per_user
  

def Simulate_SUSDP_GKMPS_speed(values):
    gkmps = init_GKMPS()

    n_users = num_users()
    n_rounds = times()
    k_malicious = k()
    U = gkmps.U

    # ===== Pre-compute constant terms =====
    exp_eps_star = math.exp(-gkmps.epsilonstar / U)
    exp_eps1 = math.exp(-0.2 * gkmps.epsilon1 / U)
    log_delta1 = 3 * (1 + math.log(1 / gkmps.delta1))
    log_delta2_factor = 3 * (1 + math.log((2 * U - 1) / gkmps.delta2))

    # ===== Parameters for random sampling =====
    p_plus_minus = 1 - exp_eps_star
    p_eps1 = 1 - exp_eps1
    p_eps2 = np.array([1 - math.exp(-0.1 * gkmps.epsilon2 / gkmps.t[i])
                       for i in range(-U, U + 1)])

    errors = np.zeros(n_rounds)
    dp_sums = np.zeros(n_rounds)
    nmessages_per_user = np.zeros(n_rounds)

    malicious_all = sorted_malicious()  # Cache the malicious user indices for all rounds

    for r in range(n_rounds):
        vals = np.array(values[r])
        honest_mask = np.ones(n_users, dtype=bool)
        honest_mask[malicious_all[r]] = False
        honest_vals = vals[honest_mask]

        # ---------- Vectorized Randomized Rounding ----------
        rvalues = np.array([gkmps.RandomizedRounding(v) for v in honest_vals])
        nonzero_mask = (rvalues != 0)
        nmessages = np.count_nonzero(nonzero_mask)
        dp_sum = np.sum(rvalues)

        # ---------- Central-DP noise (Negative Binomial) ----------
        # Use NumPy vectorization for batch sampling
        num_plus_1 = np.random.negative_binomial(1.0, p_plus_minus, size=len(honest_vals))
        num_minus_1 = np.random.negative_binomial(1.0, p_plus_minus, size=len(honest_vals))
        nmessages += np.sum(num_plus_1 + num_minus_1)
        dp_sum += np.sum(num_plus_1 - num_minus_1)

        # ---------- Zero-sum DP noise generation ----------
        # Generate all noise values in a single vectorized step
        z = np.zeros(2 * U + 1)
        idxs = np.arange(-U, U + 1)
        mask_nonzero = (idxs != 0) & (idxs != -1)

        # Handle i = 1 and other i separately
        z[U + 1] += np.random.negative_binomial(log_delta1, p_eps1)
        z[mask_nonzero] += np.random.negative_binomial(
            log_delta2_factor,
            p_eps2[mask_nonzero],
            size=np.count_nonzero(mask_nonzero)
        )

        # ---------- Update message count ----------
        nmessages += 2 * int(z[U + 1]) + 3 * np.sum(z[mask_nonzero & (idxs != 1)])

        # ---------- Record results ----------
        dp_sums[r] = dp_sum
        errors[r] = abs(dp_sum - np.sum(vals))
        nmessages_per_user[r] = nmessages / (n_users - k_malicious)

    return dp_sums, errors, nmessages_per_user
