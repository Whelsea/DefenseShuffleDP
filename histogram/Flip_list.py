import os
import random
import numpy as np
import pandas as pd
import math
import time
from typing import Union, List, Dict, Tuple


class PFLIPSimulator:
    def __init__(self, d: int, epsilon: float, delta: float, n: int, k: int = None):
        self.d = d
        self.epsilon = epsilon
        self.delta = delta
        self.n = n

        # Calculate minimum k based on Theorem 3.3
        min_k = self.calculate_min_k()

        # Use provided k if it's sufficient, otherwise use calculated minimum
        if k is None or k < min_k:
            self.k = math.ceil(min_k)
            print(f"Using calculated k = {self.k}")
        else:
            self.k = k
            print(f"Using provided k = {self.k}")

        # Calculate q based on k according to Claim 3.2
        self.q = self.calculate_q()
        print(f"Calculated q = {self.q:.6f}")

        # Calculate scaling factor f(k) = 1/(1-2q)
        self.scaling_factor = 1 / (1 - 2 * self.q)
        print(f"Scaling factor f(k) = {self.scaling_factor:.6f}")

    def calculate_min_k(self) -> float:
        """
        Calculate minimum value of k based on Theorem 3.3
        """
        e_eps = math.exp(self.epsilon)
        term = ((e_eps + 1) / (e_eps - 1)) ** 2
        return (132 / (5 * self.n)) * term * math.log(4 / self.delta)

    def calculate_q(self) -> float:
        """
        Calculate q based on Claim 3.2: q(1-q) ≥ (33/(5nk)) * ((e^ε+1)/(e^ε-1))^2 * ln(4/δ)
        """
        e_eps = math.exp(self.epsilon)
        term = ((e_eps + 1) / (e_eps - 1)) ** 2
        C = (33 / (5 * self.n * self.k)) * term * math.log(4 / self.delta)

        # Ensure the constraint is satisfiable (C ≤ 0.25 for q < 0.5)
        if C > 0.25:
            raise ValueError(f"Cannot satisfy privacy constraint with current parameters. Try increasing k.")

        # Solve q(1-q) = C, taking the solution where q < 0.5
        q = 0.5 - math.sqrt(0.25 - C)

        return q

    def get_expected_error(self) -> float:
        bias = (math.exp(self.epsilon) + 1) / (math.exp(self.epsilon) - 1)
        tail = math.sqrt((132 / 5) * math.log(4 / self.delta) *
                         math.log2(20 * self.d))  # 用 log2
        scale = math.sqrt(self.k + 1) / self.n  # 加 √(k+1)/n
        return scale * bias * tail * self.scaling_factor

    def simulate(self, data: Union[List[int], np.ndarray, Dict[int, int]]) -> np.ndarray:
        """
        Simulate the PFLIP protocol without explicitly generating messages.

        Input:
            data: Either a list of indices (one per user), a histogram, or a dict mapping indices to counts

        Output:
            Estimated histogram
        """
        # Convert input data to a histogram
        if isinstance(data, dict):
            # Dictionary format {index: count}
            true_histogram = np.zeros(self.d)
            for idx, count in data.items():
                true_histogram[idx] = count / self.n
        elif isinstance(data, np.ndarray) and len(data.shape) == 1 and data.shape[0] == self.d:
            # Already a histogram
            true_histogram = data.copy() / self.n if np.sum(data) > self.n else data.copy()
        else:
            # List of indices
            true_histogram = np.zeros(self.d)
            for idx in data:
                true_histogram[idx] += 1
            true_histogram /= self.n

        # Simulate the protocol outcome directly
        estimated_histogram = self._simulate_outcome(true_histogram)

        return estimated_histogram

    def _simulate_outcome(self, true_histogram: np.ndarray) -> np.ndarray:
        """
        Simulate the outcome of the protocol given the true histogram.

        Args:
            true_histogram: True normalized histogram (sums to 1)

        Returns:
            Estimated histogram
        """
        # Step 1: Calculate the expected counts after randomized response
        expected_counts = np.zeros(self.d)
        for j in range(self.d):
            expected_counts[j] = (
                    true_histogram[j] * (1 - self.q) +  # True 1s that stay 1
                    (1 - true_histogram[j]) * self.q  # True 0s that become 1
            )

        # Step 2: Add noise to simulate variation
        noisy_counts = np.zeros(self.d)
        for j in range(self.d):
            p = expected_counts[j]
            # std_dev = math.sqrt(self.n * p * (1 - p))
            # noisy_counts[j] = np.random.normal(self.n * p, std_dev)
            noisy_counts[j] = np.random.binomial(self.n, p)

        # Step 3: Account for fake users
        fake_user_count = self.k * self.n

        for j in range(self.d):
            # std_dev = math.sqrt(fake_user_count * self.q * (1 - self.q))
            # fake_noise = np.random.normal(fake_user_count * self.q, std_dev)
            # noisy_counts[j] += fake_noise
            fake_noise = np.random.binomial(fake_user_count, self.q)
            noisy_counts[j] += fake_noise

        # Step 4: De-bias and scale
        total_messages = self.n + fake_user_count
        estimated_histogram = np.zeros(self.d)
        for j in range(self.d):
            estimated_histogram[j] = (1 / self.n) * self.scaling_factor * (noisy_counts[j] - self.q * total_messages)

        return estimated_histogram

    def theoretical_variance(self) -> float:
        """
        Calculate the theoretical variance of each bin estimate based on Claim 3.6
        """
        variance = (self.k + 1) / self.n * self.q * (1 - self.q) * (self.scaling_factor ** 2)
        return variance

    def get_confidence_interval(self, confidence: float = 0.9) -> float:
        """
        Get the confidence interval width for a single bin at given confidence level

        Args:
            confidence: Confidence level (e.g., 0.9 for 90% CI)

        Returns:
            Width of confidence interval for a single bin
        """
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence) / 2)
        std_dev = math.sqrt(self.theoretical_variance())
        return z_score * std_dev

    def estimate_message_complexity(self) -> float:
        """
        Estimate the average number of messages per user
        """
        return self.k + 1


def loaddata(path: str) -> tuple[list[int], int, int]:
    with open(path, 'r') as f:
        n = int(f.readline())
        B = int(f.readline())
        return [int(f.readline()) for _ in range(n)], n, B


def calculate_error_metrics(true_histogram, estimated_histogram):
    # Calculate absolute errors for all bins
    abs_errors = np.abs(true_histogram - estimated_histogram)

    # Calculate non-zero indices
    non_zero_indices = np.where(true_histogram > 0)[0]
    abs_errors_non_zero = abs_errors[non_zero_indices]

    # Calculate different error metrics
    metrics = {
        "linf_error": np.max(abs_errors),
        "linf_error_non_zero": np.max(abs_errors_non_zero) if len(abs_errors_non_zero) > 0 else 0,
        "p50_error": np.percentile(abs_errors, 50),
        "p90_error": np.percentile(abs_errors, 90),
        "p95_error": np.percentile(abs_errors, 95),
        "p99_error": np.percentile(abs_errors, 99)
    }

    return metrics


def generate_results_text(simulator, true_histogram, all_metrics, simulation_times, expected_error, ci_width,
                          non_zero_samples):
    """
    Generate a formatted text of simulation results with multiple runs
    Returns:
        String containing all the results text
    """
    output_text = []

    # Calculate average metrics
    avg_metrics = {}
    for metric in all_metrics[0].keys():
        metric_values = [run_metrics[metric] for run_metrics in all_metrics]
        trim_count = int(0.1 * len(all_metrics))  # 计算 10% 的数量

        # 确保去掉的数量不会导致没有剩余数据
        if trim_count * 2 < n:
            sorted_values = sorted(metric_values)
            trimmed_values = sorted_values[trim_count:-trim_count]  # 去掉前 10% 和后 10%
            avg_metrics[metric] = np.mean(trimmed_values)
        else:
            # 如果数据太少，直接计算平均值
            avg_metrics[metric] = np.mean(metric_values)

        # avg_metrics[metric] = np.mean(metric_values)

    avg_simulation_time = np.mean(simulation_times)

    # Add general results
    output_text.append("\nResults:")
    output_text.append(f"Number of runs: {len(all_metrics)}")
    output_text.append(f"Average simulation time: {avg_simulation_time:.2f} seconds")
    output_text.append(f"Linf error = {avg_metrics['linf_error']:.6f}")
    output_text.append(f"50% error = {avg_metrics['p50_error']:.6f}")
    output_text.append(f"90% error = {avg_metrics['p90_error']:.6f}")
    output_text.append(f"95% error = {avg_metrics['p95_error']:.6f}")
    output_text.append(f"99% error = {avg_metrics['p99_error']:.6f}")
    output_text.append(f"Expected max error (theory): {expected_error:.6f}")
    output_text.append(f"90% confidence interval width: {ci_width:.6f}")
    output_text.append(f"# Messages per user: {simulator.estimate_message_complexity()}")
    output_text.append(f"# Bits per message: {simulator.d}")

    # Add detailed metrics from each run
    output_text.append("\nDetailed Results:")
    output_text.append("Run\tTime(s)\tLinf\t50%\t90%\t95%\t99%")
    for i in range(len(all_metrics)):
        metrics = all_metrics[i]
        output_text.append(
            f"{i + 1}\t{simulation_times[i]:.2f}\t{metrics['linf_error']:.6f}\t{metrics['p50_error']:.6f}\t{metrics['p90_error']:.6f}\t{metrics['p95_error']:.6f}\t{metrics['p99_error']:.6f}")

    # Add arrays of values
    output_text.append("\nError Arrays:")

    # Linf errors
    linf_errors = [metrics['linf_error'] for metrics in all_metrics]
    output_text.append(f"Linf errors: {linf_errors}")

    # 50% errors
    p50_errors = [metrics['p50_error'] for metrics in all_metrics]
    output_text.append(f"50% errors: {p50_errors}")

    # 90% errors
    p90_errors = [metrics['p90_error'] for metrics in all_metrics]
    output_text.append(f"90% errors: {p90_errors}")

    # 95% errors
    p95_errors = [metrics['p95_error'] for metrics in all_metrics]
    output_text.append(f"95% errors: {p95_errors}")

    # 99% errors
    p99_errors = [metrics['p99_error'] for metrics in all_metrics]
    output_text.append(f"99% errors: {p99_errors}")

    # Add sample histogram entries (from the last run)
    if non_zero_samples:
        output_text.append("\nSample of true vs estimated histogram (from last run):")
        for idx, true_val, est_val, diff in non_zero_samples:
            output_text.append(f"Index {idx}: True={true_val:.6f}, Est={est_val:.6f}, Diff={diff:.6f}")

    return "\n".join(output_text)


def load_sf_salary_for_frequency(n: int, d: int, seed: int = 42) -> list[int]:
    """
    读取 SF_Salaries 数据，提取 BasePay 并映射到整数域 [0, d-1]
    Args:
        n: 需要的样本数量
        d: 频率统计的域大小
        seed: 随机种子
    Returns:
        一个长度为 n 的整数列表，范围为 [0, d-1]
    """

    path = "./data/Salary/SF_Salaries/data.csv"
    df = pd.read_csv(path)

    salary = pd.to_numeric(df['BasePay'], errors='coerce')
    salary = salary.fillna(0).clip(lower=0)

    # 映射：线性缩放到 [0, d-1]
    salary = salary[salary > 0]
    max_val = salary.quantile(0.999)  # 去除极端值（极大值不参与缩放）
    scaled = (salary / max_val * (d - 1)).clip(0, d - 1).astype(int)

    # 截断或扩充
    if len(scaled) >= n:
        result = scaled.sample(n=n, random_state=seed).tolist()
    else:
        result = scaled.tolist()
        extra = random.choices(result, k=n - len(result))
        result += extra

    return result


import time

# Main execution
if __name__ == "__main__":
    # Parameters
    epsilon = 4.0
    k = 2
    times = 50  # Number of times to run the simulation
    attacker_num = 1
    dataset = "Simulated data"
    # dataset = "realworld data"
    # list_distribution = ["Unif", "Gauss", "Zip"]
    list_distribution = ["Gauss"]
    list_realdata = ["aol_data", "twitter_data", "SF_Sal"]
    # dataset = "aol dataset"
    # Load data
    for n in {131072}:
        for d in {131072}:
            # for n in {3700000}:
            #     for d in {470000}:
            b = math.log2(d)
            if dataset == "Simulated data":

                #     true_values, n, d = loaddata(f"./data/aol_data/trans_01_n_{n}_b_{int(b)}_B_{d}.txt")
                # else:
                #     true_values, n, d = loaddata(f"./data/Gauss/Gauss_n{n}B{d}")
                #     # true_values=np.random.randint(0, d, size=n)
                for distribution in list_distribution:
                    if distribution == "Unif":
                        true_values = np.random.randint(0, d, size=n)
                    else:
                        true_values, n, d = loaddata(f"./data/{distribution}/{distribution}_n{n}B{d}")
                    delta = 1.0 / (n ** 2)
                    # delta = 0.0000001

                    malicious_users = []
                    if attacker_num > 0:
                        malicious_users = set(random.sample(range(n), attacker_num))
                    sorted_malicious = sorted(malicious_users)

                    # Calculate true histogram
                    true_histogram = np.zeros(d)
                    for v in true_values:
                        true_histogram[v] += 1

                    true_histogram = true_histogram / n

                    # Create arrays to store metrics from each run
                    all_metrics = []
                    simulation_times = []

                    # Initialize the simulator (done only once to avoid repeating setup output)
                    simulator = PFLIPSimulator(d, epsilon, delta, n, k)
                    expected_error = simulator.get_expected_error()
                    ci_width = simulator.get_confidence_interval(0.9)

                    # Run the simulation multiple times
                    print(f"Running simulation {times} times...")
                    for t in range(times):
                        print(f"Run {t + 1}/{times}")

                        # # Set a different random seed for each run
                        # np.random.seed(42 + t)

                        # Run a single simulation
                        start_time = time.time()
                        estimated_histogram = simulator.simulate(true_values)
                        end_time = time.time()

                        # Simulate attacker behavior: k noisy message
                        for i in sorted_malicious:
                            noisy_bin = np.random.randint(0, d)
                            estimated_histogram[noisy_bin] += k / n

                        # Calculate metrics for this run
                        metrics = calculate_error_metrics(true_histogram, estimated_histogram)
                        simulation_time = end_time - start_time

                        # Store metrics
                        all_metrics.append(metrics)
                        simulation_times.append(simulation_time)

                    # Save sample indices and values from the last run
                    non_zero_samples = []
                    non_zero_indices = np.where(true_histogram > 0)[0]
                    sample_indices = np.random.choice(non_zero_indices, size=min(5, len(non_zero_indices)),
                                                      replace=False)
                    for idx in sample_indices:
                        true_val = true_histogram[idx]
                        est_val = estimated_histogram[idx]
                        diff = abs(true_val - est_val)
                        non_zero_samples.append((idx, true_val, est_val, diff))

                    # Generate results text
                    results_text = generate_results_text(
                        simulator,
                        true_histogram,
                        all_metrics,
                        simulation_times,
                        expected_error,
                        ci_width,
                        non_zero_samples
                    )

                    # Write results to file
                    result_root = os.path.join(".", "Result")
                    baseline_folder = os.path.join(result_root, "Flip")
                    if not os.path.exists(baseline_folder):
                        os.makedirs(baseline_folder)
                    if dataset == "aol dataset":
                        data_folder = os.path.join(baseline_folder, "aol_data")
                    else:
                        data_folder = os.path.join(baseline_folder, f"simulated_data/{distribution}")
                    if not os.path.exists(data_folder):
                        os.makedirs(data_folder)
                    outfile_name = f"FLip_n{n}_B{d}_attacker{attacker_num}_k{k}_eps{epsilon}.txt"
                    outfile_path = os.path.join(data_folder, outfile_name)

                    with open(outfile_path, "w") as f:
                        f.write(results_text)
            else:
                for dataset_name in list_realdata:
                    if dataset_name == "aol_data":
                        b = math.log2(d)
                        true_values, n, d = loaddata(f"./data/aol_data/trans_01_n_{n}_b_{int(b)}_B_{d}.txt")
                    elif dataset_name == "twitter_data":
                        true_values, n, d = loaddata(f"./data/twitter_data/twitwer_n131072B131072.txt")
                    elif dataset_name == "SF_Sal":
                        true_values = load_sf_salary_for_frequency(n, d)
                    delta = 1.0 / (n ** 2)

                    malicious_users = []
                    if attacker_num > 0:
                        malicious_users = set(random.sample(range(n), attacker_num))
                    sorted_malicious = sorted(malicious_users)

                    # Calculate true histogram
                    true_histogram = np.zeros(d)
                    for v in true_values:
                        true_histogram[v] += 1

                    true_histogram = true_histogram / n

                    # Create arrays to store metrics from each run
                    all_metrics = []
                    simulation_times = []

                    # Initialize the simulator (done only once to avoid repeating setup output)
                    simulator = PFLIPSimulator(d, epsilon, delta, n, k)
                    expected_error = simulator.get_expected_error()
                    ci_width = simulator.get_confidence_interval(0.9)

                    # Run the simulation multiple times
                    print(f"Running simulation {times} times...")
                    for t in range(times):
                        print(f"Run {t + 1}/{times}")

                        # # Set a different random seed for each run
                        # np.random.seed(42 + t)

                        # Run a single simulation
                        start_time = time.time()
                        estimated_histogram = simulator.simulate(true_values)
                        end_time = time.time()

                        # Simulate attacker behavior: k noisy message
                        for i in sorted_malicious:
                            noisy_bin = np.random.randint(0, d)
                            estimated_histogram[noisy_bin] += k / n

                        # Calculate metrics for this run
                        metrics = calculate_error_metrics(true_histogram, estimated_histogram)
                        simulation_time = end_time - start_time

                        # Store metrics
                        all_metrics.append(metrics)
                        simulation_times.append(simulation_time)

                    # Save sample indices and values from the last run
                    non_zero_samples = []
                    non_zero_indices = np.where(true_histogram > 0)[0]
                    sample_indices = np.random.choice(non_zero_indices, size=min(5, len(non_zero_indices)),
                                                      replace=False)
                    for idx in sample_indices:
                        true_val = true_histogram[idx]
                        est_val = estimated_histogram[idx]
                        diff = abs(true_val - est_val)
                        non_zero_samples.append((idx, true_val, est_val, diff))

                    # Generate results text
                    results_text = generate_results_text(
                        simulator,
                        true_histogram,
                        all_metrics,
                        simulation_times,
                        expected_error,
                        ci_width,
                        non_zero_samples
                    )

                    # Write results to file
                    result_root = os.path.join(".", "Result")
                    baseline_folder = os.path.join(result_root, "Flip")
                    if not os.path.exists(baseline_folder):
                        os.makedirs(baseline_folder)
                    data_folder = os.path.join(baseline_folder, f"{dataset_name}")
                    if not os.path.exists(data_folder):
                        os.makedirs(data_folder)
                    outfile_name = f"FLip_n{n}_B{d}_attacker{attacker_num}_k{k}_eps{epsilon}.txt"
                    outfile_path = os.path.join(data_folder, outfile_name)

                    with open(outfile_path, "w") as f:
                        f.write(results_text)
