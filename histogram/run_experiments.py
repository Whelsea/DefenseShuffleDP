import os
import sys
import time
import numpy as np
import pandas as pd
import random
import math
from typing import List, Tuple
import FE1_Simulator
import simulate_ours_fe
import Flip_list


def load_sf_salary_for_frequency(n: int, d: int, seed: int = 42) -> list[int]:
    """
    Load SF_Salaries 'BasePay' column, mapping to [0, d-1]
    """

    path = "./data/SF_Salaries/data.csv"
    df = pd.read_csv(path)

    salary = pd.to_numeric(df['BasePay'], errors='coerce')
    salary = salary.fillna(0).clip(lower=0)

    salary = salary[salary > 0]
    max_val = salary.quantile(0.999) 
    scaled = (salary / max_val * (d - 1)).clip(0, d - 1).astype(int)

    if len(scaled) >= n:
        result = scaled.sample(n=n, random_state=seed).tolist()
    else:
        result = scaled.tolist()
        extra = random.choices(result, k=n - len(result))
        result += extra

    return result


def load_br_salary_for_frequency(n: int, d: int, seed: int = 42) -> list[int]:
    """
    Load BR_Salaries 'BasePay' column, mapping to [0, d-1]
    """

    path = "./data/BR_Salaries/data.csv"
    df = pd.read_csv(path)

    salary = pd.to_numeric(df['total_salary'], errors='coerce')
    salary = salary.fillna(0).clip(lower=0)

    salary = salary[salary > 0]
    max_val = salary.quantile(0.999) 
    scaled = (salary / max_val * (d - 1)).clip(0, d - 1).astype(int)

    if len(scaled) >= n:
        result = scaled.sample(n=n, random_state=seed).tolist()
    else:
        result = scaled.tolist()
        extra = random.choices(result, k=n - len(result))
        result += extra

    return result


def load_data_by_mode(data_mode: str, n: int, B: int, seed: int = 42) -> np.ndarray:
    """
    Unified data loading function
    """
    if data_mode == "zipf":
        path = f"./data/Zip/Zip_n{n}B{B}"
        with open(path, 'r') as f:
            n_file = int(f.readline())
            B_file = int(f.readline())
            data = [int(f.readline()) for _ in range(n_file)]
        return np.array(data)

    elif data_mode == "gauss":
        path = f"./data/Gauss/Gauss_n{n}B{B}"
        with open(path, 'r') as f:
            n_file = int(f.readline())
            B_file = int(f.readline())
            data = [int(f.readline()) for _ in range(n_file)]
        return np.array(data)

    elif data_mode == "aol":
        b = int(math.log2(B))
        path = f"./data/aol_data/trans_01_n_{n}_b_{b}_B_{B}.txt"
        with open(path, 'r') as f:
            n_file = int(f.readline())
            B_file = int(f.readline())
            data = [int(f.readline()) for _ in range(n_file)]
        return np.array(data)

    elif data_mode == "unif":
        np.random.seed(seed)
        return np.random.randint(1, B, size=n)

    elif data_mode == "twitter":
        path = f"./data/twitter_data/twitwer_n{n}B{B}.txt"
        with open(path, 'r') as f:
            n_file = int(f.readline())
            B_file = int(f.readline())
            data = [int(f.readline()) for _ in range(n_file)]
        return np.array(data)

    elif data_mode == "sf_sal":
        return np.array(load_sf_salary_for_frequency(n, B, seed))

    elif data_mode == "br_sal":
        return np.array(load_sf_salary_for_frequency(n, B, seed))


    else:
        raise ValueError(f"Unsupported data_mode: {data_mode}")


def run_fe1_algorithm(data: np.ndarray, n: int, B: int, epsilon: float, delta: float,
                      c: float, beta: float, k: int = 0, num_runs: int = 50, trim_count: int = 5) -> dict:
    """Run FE1(LWY)"""
    sim = FE1_Simulator.FE1Simulator(n, B, epsilon, delta, c, beta)

    # real freq
    true_freq = np.zeros(B + 1)
    for x in data:
        true_freq[x] += 1

    results = {'runtime': [], 'max_error': [], 'msg_count': [], 'bit_count': []}

    print(f"Running FE1 with {num_runs} iterations...")
    for run in range(num_runs):
        print(f"  Run {run + 1}/{num_runs}")

        start_time = time.time()
        est_freq = sim.simulate_parallel(data, 1)
        if k > 0:
            est_freq += k * n
        runtime = time.time() - start_time

        errors = np.abs(true_freq[1:] - est_freq[1:])
        max_error = np.max(errors)

        results['runtime'].append(runtime)
        results['max_error'].append(max_error)
        results['msg_count'].append(1 + sim.sample_prob)
        results['bit_count'].append(math.ceil(math.log2(sim.q)) * 2 + math.ceil(math.log2(sim.b)))

    def trimmed_mean(data, trim):
        sorted_data = sorted(data)
        return np.mean(sorted_data[trim:-trim] if trim > 0 else sorted_data)

    return {
        'mu': sim.mu,
        'runtime': trimmed_mean(results['runtime'], trim_count),
        'max_error': trimmed_mean(results['max_error'], trim_count),
        'msg_count': trimmed_mean(results['msg_count'], trim_count),
        'bit_count': trimmed_mean(results['bit_count'], trim_count)
    }


def run_ours_fe_algorithm(data: np.ndarray, n: int, B: int, epsilon: float, delta: float,
                          c: float, lambda_n, beta: float, k: int = 0, num_runs: int = 50, trim_count: int = 5) -> dict:
    """run ours+FE1(LWY)"""
    # set global parameters
    simulate_ours_fe.num_users = n
    simulate_ours_fe.domain = B
    simulate_ours_fe.epsilon = epsilon
    simulate_ours_fe.delta = delta
    simulate_ours_fe.k = k
    simulate_ours_fe.beta = beta
    simulate_ours_fe.d = B
    simulate_ours_fe.C = c
    simulate_ours_fe.custom_lambda_n = lambda_n

    # corrupted users
    malicious_users = []
    if k > 0:
        malicious_users = set(random.sample(range(n), k))
    simulate_ours_fe.malicious_users = malicious_users
    simulate_ours_fe.sorted_malicious = sorted(malicious_users)

    # true freq
    true_freq = np.zeros(B + 1)
    for x in data:
        true_freq[x] += 1

    results = {'max_error': [], 'l50_error': [], 'l90_error': [], 'l95_error': [], 'l99_error': [], 'msg_count': []}

    print(f"Running Ours+FE1 with {num_runs} iterations...")
    for run in range(num_runs):
        print(f"  Run {run + 1}/{num_runs}")

        est_freq, nmessages_per_user = simulate_ours_fe.simulate_ours_FE1(data.tolist())
        errors = np.abs(true_freq[1:] - est_freq[1:])
        max_error = np.max(errors)

        sorted_errors = sorted(errors)
        l50_error = sorted_errors[int(0.50 * B)]
        l90_error = sorted_errors[int(0.90 * B)]
        l95_error = sorted_errors[int(0.95 * B)]
        l99_error = sorted_errors[int(0.99 * B)]

        results['max_error'].append(max_error)
        results['l50_error'].append(l50_error)
        results['l90_error'].append(l90_error)
        results['l95_error'].append(l95_error)
        results['l99_error'].append(l99_error)
        results['msg_count'].append(nmessages_per_user)

    def trimmed_mean(data, trim):
        sorted_data = sorted(data)
        return np.mean(sorted_data[trim:-trim] if trim > 0 else sorted_data)

    return {
        'lambda': simulate_ours_fe.find_lambda(n),
        'max_error': trimmed_mean(results['max_error'], trim_count),
        'l50_error': trimmed_mean(results['l50_error'], trim_count),
        'l90_error': trimmed_mean(results['l90_error'], trim_count),
        'l95_error': trimmed_mean(results['l95_error'], trim_count),
        'l99_error': trimmed_mean(results['l99_error'], trim_count),
        'msg_count': trimmed_mean(results['msg_count'], trim_count)
    }


def save_results(results: dict, algorithm_name: str, data_mode: str, n: int, B: int,
                 lambda_n, epsilon: float, c: float, k: int = 0, additional_params: dict = None):
    """save results to files"""
    result_dir = os.path.join(".", f"Result", algorithm_name, data_mode.capitalize())
    os.makedirs(result_dir, exist_ok=True)

    if additional_params:
        param_str = "_".join([f"{key}{val}" for key, val in additional_params.items()])
        result_file = os.path.join(result_dir,
                                   f"{data_mode}_n{n}B{B}_eps{epsilon}_c{c}_k{k}_lam{lambda_n}_{param_str}.txt")
    else:
        result_file = os.path.join(result_dir, f"{data_mode}_n{n}B{B}_eps{epsilon}_c{c}_k{k}_lam{lambda_n}.txt")

    with open(result_file, 'w') as f:
        f.write(f"=== {algorithm_name} Results ({data_mode.upper()} - n={n}, B={B}) ===\n")
        f.write(f"Epsilon: {epsilon}\n")
        f.write(f"C parameter: {c}\n")
        f.write(f"Attackers: {k}\n\n")

        for key, value in results.items():
            f.write(f"{key}: {value}\n")

    print(f"Results saved to: {os.path.abspath(result_file)}")
    return result_file


def run_flip_algorithm(data: np.ndarray, n: int, B: int, epsilon: float, delta: float,
                       k_flip: int = 2, attacker_num: int = 0,
                       num_runs: int = 50, trim_count: int = 5) -> dict:
    """run Flip(CZ)"""

    # true freq
    true_histogram = np.zeros(B)
    for x in data:
        true_histogram[x] += 1
    true_histogram = true_histogram / n

    malicious_users = []
    if attacker_num > 0:
        malicious_users = set(random.sample(range(n), attacker_num))
    sorted_malicious = sorted(malicious_users)

    simulator = Flip_list.PFLIPSimulator(B, epsilon, delta, n, k_flip)

    results = {
        'runtime': [], 'linf_error': [], 'p50_error': [], 'p90_error': [],
        'p95_error': [], 'p99_error': [], 'msg_count': [], 'bit_count': []
    }

    print(f"Running PFLIP with {num_runs} iterations...")
    for run in range(num_runs):
        print(f"  Run {run + 1}/{num_runs}")

        start_time = time.time()
        estimated_histogram = simulator.simulate(data.tolist())
        runtime = time.time() - start_time

        for i in sorted_malicious:
            noisy_bin = np.random.randint(0, B)
            estimated_histogram[noisy_bin] += k_flip / n

        metrics = Flip_list.calculate_error_metrics(true_histogram, estimated_histogram)

        results['runtime'].append(runtime)
        results['linf_error'].append(metrics['linf_error'])
        results['p50_error'].append(metrics['p50_error'])
        results['p90_error'].append(metrics['p90_error'])
        results['p95_error'].append(metrics['p95_error'])
        results['p99_error'].append(metrics['p99_error'])
        results['msg_count'].append(simulator.estimate_message_complexity())
        results['bit_count'].append(B)

    def trimmed_mean(data, trim):
        sorted_data = sorted(data)
        return np.mean(sorted_data[trim:-trim] if trim > 0 else sorted_data)

    return {
        'k_flip': simulator.k,
        'q': simulator.q,
        'scaling_factor': simulator.scaling_factor,
        'runtime': trimmed_mean(results['runtime'], trim_count),
        'linf_error': trimmed_mean(results['linf_error'], trim_count),
        'p50_error': trimmed_mean(results['p50_error'], trim_count),
        'p90_error': trimmed_mean(results['p90_error'], trim_count),
        'p95_error': trimmed_mean(results['p95_error'], trim_count),
        'p99_error': trimmed_mean(results['p99_error'], trim_count),
        'msg_count': trimmed_mean(results['msg_count'], trim_count),
        'bit_count': trimmed_mean(results['bit_count'], trim_count),
        'theoretical_variance': simulator.theoretical_variance()
    }


def main():
    # parameter setting
    # algorithms = ["Flip", "FE1", "Ours+FE1"]
    algorithms = ["Flip"]
    data_modes = ["aol"]  # ["unif", "zipf", "gauss", "aol", "sf_salary"]
    # list_n = [2 ** 12, 2 ** 16, 2 ** 20, 2 ** 24]
    list_n = [2 ** 17]
    list_B = [131072]
    list_epsilon = [4]
    list_c = [1.0]
    list_k = [1]
    list_lambda = [8192]
    # fixed parameters
    fixed_beta = 0.1
    fixed_num_runs = 10
    fixed_trim_count = 1

    print("=" * 60)
    print("Starting Frequency Estimation Algorithm Comparison")
    print("=" * 60)

    for data_mode in data_modes:
        print(f"\nProcessing dataset: {data_mode.upper()}")

        for n in list_n:
            for lambda_n in list_lambda:
                for B in list_B:
                    print(f"\n Loading data: n={n}, B={B}")

                    try:
                        data = load_data_by_mode(data_mode, n, B)
                        print(f"✅ Data loaded successfully. Shape: {data.shape}")

                        for epsilon in list_epsilon:
                            delta = 1 / (n * n)

                            for c in list_c:
                                for k in list_k:
                                    print(f"\nRunning experiments: ε={epsilon}, c={c}, k={k}")

                                    for algorithm in algorithms:
                                        print(f"\nAlgorithm: {algorithm}")

                                        try:
                                            if algorithm == "FE1":
                                                results = run_fe1_algorithm(
                                                    data, n, B, epsilon, delta, c, fixed_beta,
                                                    k, fixed_num_runs, fixed_trim_count
                                                )

                                            elif algorithm == "Ours+FE1":
                                                results = run_ours_fe_algorithm(
                                                    data, n, B, epsilon, delta, c, lambda_n, fixed_beta, k,
                                                    fixed_num_runs, fixed_trim_count
                                                )

                                            elif algorithm == "Flip":
                                                results = run_flip_algorithm(
                                                    data, n, B, epsilon, delta, 2, k, fixed_num_runs, fixed_trim_count
                                                )

                                            # 保存结果
                                            save_results(
                                                results, algorithm, data_mode, n, B, lambda_n, epsilon, c, k,
                                                {"runs": fixed_num_runs, "trim": fixed_trim_count}
                                            )

                                            print(f"✅ {algorithm} completed successfully")

                                        except Exception as e:
                                            print(f" Error running {algorithm}: {str(e)}")
                                            continue

                    except Exception as e:
                        print(f"Error loading data for {data_mode}: {str(e)}")
                        continue

    print("\n" + "=" * 60)
    print("All experiments completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
