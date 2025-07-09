# -*- coding: utf-8 -*-

import os
import sys
import time
import numpy as np
import pandas as pd
import advanced_HSDP
import random
import math


# Read adult age data
def read_and_clip_age(file_path):
    values = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            try:
                # Split each line of data
                parts = line.split(', ')
                # Extract first column (age) and convert to integer
                age_str = parts[0].strip()
                if not age_str:  # If age field is empty
                    continue

                age = int(age_str)
                # Clip to 130 if age exceeds 130
                if age > 130:
                    age = 130
                values.append(age)
            except (ValueError, IndexError) as e:
                print("1")
                continue
    return values


def convert_gender(file_path):
    values = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split each line of data
            parts = line.strip().split(', ')
            if len(parts) < 9:  # Ensure enough columns exist
                continue
            # Extract gender column (9th column)
            gender = parts[9].strip()  # Remove leading/trailing spaces
            # Female=1, Male=0
            if gender.lower() == 'female':
                values.append(1)
            elif gender.lower() == 'male':
                values.append(0)
    return values


def load_dataset(problem, dataset_name):
    if dataset_name == "Adult":
        if problem == "Bit Counting":
            value = convert_gender("./data/adult.data")
            extended_value = value + random.choices(value, k=32768 - len(value))
            n = len(extended_value)
            d = 2
            return np.array(extended_value), n, d
        elif problem == "Summation":
            value = read_and_clip_age("./data/adult.data")
            extended_value = value + random.choices(value, k=32768 - len(value))
            n = len(extended_value)
            d = 131
            return np.array(extended_value), n, d
    else:
        data = pd.read_csv(f"./data/Salary/{dataset_name}/data.csv")
        if dataset_name == "BR_Salaries":
            salary_column = 'total_salary'
        elif dataset_name == "SF_Salaries":
            salary_column = 'BasePay'
        elif dataset_name == "Ont_Salaries":
            salary_column = 'Salary Paid'
        salary = pd.to_numeric(data[salary_column], errors='coerce').fillna(0)
        trimmed_len = 2 ** math.floor(math.log2(len(salary)))
        salary = salary.iloc[:trimmed_len].reset_index(drop=True)

        if problem == "Bit Counting":
            threshold = salary.mean()
            binary_value = np.where(salary > threshold, 1, 0).astype(int)
            n = len(binary_value)
            d = 2
            return binary_value, n, d
        elif problem == "Summation":
            n = len(salary)
            d = int(2.5 * 10 ** 5 + 1)
            clipped_salary = salary.clip(lower=0, upper=d - 1).to_numpy()
            return clipped_salary, n, d


def generate_data(distribution, n, d):
    if distribution == 'Unif':
        return np.random.randint(0, d, size=n)

    elif distribution == 'Zipf':
        a = 1.5
        data = np.random.zipf(a, size=n) % d
        return data

    elif distribution == 'Gauss':

        if d == 2:
            mu = 0.2
            sigma = 0.2
            data = np.random.normal(loc=mu, scale=sigma, size=n)
            data = np.clip(np.round(data), 0, 1).astype(int)
        else:
            mu = (d - 1) / 5
            sigma = (d - 1) / 5
            data = np.random.normal(loc=mu, scale=sigma, size=n)
            data = np.clip(np.round(data), 0, d - 1).astype(int)
        return data


def main(argv):
    # Settings to iterate through
    global malicious_users
    protocols = ["simulate CSUZZ", "simulate BBGN", "simulate GKMPS", "simulate ours+BBGN", "simulate ours+GKMPS"]
    # protocols = ["simulate ours+BBGN"]
    protocol_handlers = {
        "BBGN": advanced_HSDP.baselineBBGN,
        "ours+BBGN": advanced_HSDP.ours_BBGN,
        "GKMPS": advanced_HSDP.baselineGKMPS,
        "ours+GKMPS": advanced_HSDP.ours_GKMPS,
        "simulate ours+BBGN": advanced_HSDP.simulate_ours_BBGN,
        "simulate ours+GKMPS": advanced_HSDP.simulate_ours_GKMPS,
        "simulate GKMPS": advanced_HSDP.simulateGKMPS,
        "simulate BBGN": advanced_HSDP.simulateBBGN,
        "simulate CSUZZ": advanced_HSDP.simulateCSUZZ
        # "BBGN_recursive": BBGN_recursive.run_recursive
    }
    # list_num_users = [2 ** 12, 2 ** 16, 2 ** 20, 2 ** 24]
    list_num_users = [2 ** 16]
    list_domain = [2]
    list_k = [1]
    # list_epsilon = [0.5, 1, 2, 4]
    list_epsilon = [1]
    # list_lambda = [4, 8, 32, 64, 128, 2048, 4096]
    list_lambda = [256]
    # list_dataset = ["Adult", "SF_Salaries", "Ont_Salaries", "BR_Salaries"]
    list_dataset = ["Adult"]
    list_problem = ["Bit Counting", "Summation"]
    list_distribution = ["Gauss"]
    problem = list_problem[0]

    # Fixed parameters
    fixed_epsilon = 1.0
    fixed_gamma = 0.3
    fixed_beta = 0.1
    fixed_times = 100
    fixed_sigma = 40

    # Store results in result directory
    result_root = os.path.join(".", "Result")
    data_mode = "Simulate"
    if data_mode == "Real-world":
        for dataset in list_dataset:
            value, n, d = load_dataset(problem, dataset)
            for k in list_k:
                # ================ Set parameters in advanced_HSDP.py ================
                # advanced_HSDP.custom_lambda_n = 256
                advanced_HSDP.num_users = n
                lambda_n = list_lambda[0]
                advanced_HSDP.custom_lambda_n = lambda_n
                advanced_HSDP.domain = d
                advanced_HSDP.epsilon = fixed_epsilon
                advanced_HSDP.delta = 1.0 / (n * n)
                advanced_HSDP.gamma = fixed_gamma
                advanced_HSDP.beta = fixed_beta
                advanced_HSDP.k = k
                advanced_HSDP.times = fixed_times
                advanced_HSDP.sigma = fixed_sigma
                # BBGN_recursive.times = fixed_times

                # ================ Generate values and assign to advanced_HSDP ================
                new_values = []
                sorted_malicious = []
                new_real_sums = []

                malicious_users = []
                if k > 0:
                    malicious_users = set(random.sample(range(n), k))

                for _ in range(fixed_times):
                    new_values.append(value)

                    new_real_sums.append(sum(value))
                    # ===================== Generate malicious_users =====================
                    sorted_malicious.append(sorted(malicious_users))
                advanced_HSDP.values = new_values
                advanced_HSDP.real_sums = new_real_sums
                advanced_HSDP.sorted_malicious = sorted_malicious

                # Compare different protocols on the same values dataset
                for protocol_name in protocols:
                    # Create subfolder named after baseline in Result/

                    baseline_folder = os.path.join(result_root, protocol_name)
                    if not os.path.exists(baseline_folder):
                        os.makedirs(baseline_folder)
                    problem_folder = os.path.join(baseline_folder, problem)
                    if not os.path.exists(problem_folder):
                        os.makedirs(problem_folder)
                    dataset_folder = os.path.join(problem_folder, dataset)
                    if not os.path.exists(dataset_folder):
                        os.makedirs(dataset_folder)
                    # Define output filename
                    outfile_name = f"result_n{n}_d{d}_k{k}_lam{lambda_n}_beta_split.txt"
                    outfile_path = os.path.join(dataset_folder, outfile_name)

                    print("=========================")
                    print(f"Running {protocol_name} with n={n}, domain={d} ...")
                    start_time = time.time()

                    handler = protocol_handlers.get(protocol_name)
                    dp_sums, errors, nmessages_per_user = handler(advanced_HSDP.values)

                    end_time = time.time()
                    cost_sec = end_time - start_time

                    true_sum = sum(value)
                    relative_errors = [err / true_sum for err in errors]
                    avg_relative_error = sum(np.sort(relative_errors)[10:-10]) / (len(relative_errors) - 20)

                    print(f"✅ Average Relative Error (ARE): {avg_relative_error:.6f}")

                    print(f"Done. Elapsed: {cost_sec:.4f} seconds.")

                    print("=========================\n")

                    # Write results to file
                    with open(outfile_path, 'w', encoding='utf-8') as f:
                        f.write(f"Experiment with protocol = {protocol_name}, n={n}, domain={d - 1}\n")
                        f.write(f"Elapsed time = {cost_sec:.4f} sec\n\n")
                        f.write(f"attacker id:" + str(sorted_malicious[0]) + "\n")
                        f.write("lambda: " + str(advanced_HSDP.find_lambda(n)) + "\n")
                        f.write("real sums: " + str(advanced_HSDP.real_sums) + "\n")
                        f.write("DP sums: " + str(dp_sums) + "\n")
                        f.write("Errors: " + str(errors) + "\n")
                        f.write("Average error: " + str(np.mean(np.sort(errors)[10:-10])) + "\n")
                        f.write("Average relative error: " + str(avg_relative_error) + "\n")
                        f.write("#messages per user: " + str(nmessages_per_user) + "\n")
                        f.write("Average #messages: " + str(
                            np.mean(np.sort(nmessages_per_user)[10:-10])) + "\n")

    if data_mode == "Simulate":
        # Iterate through each setting
        for n in list_num_users:
            for d in list_domain:
                for distribution in list_distribution:
                    value = generate_data(distribution, n, d)
                    for eps in list_epsilon:
                        for lambda_n in list_lambda:
                            for k in list_k:
                                # ================ Set parameters in advanced_HSDP.py ================
                                advanced_HSDP.custom_lambda_n = lambda_n
                                advanced_HSDP.num_users = n
                                advanced_HSDP.domain = d
                                advanced_HSDP.epsilon = eps
                                advanced_HSDP.delta = 1.0 / (n * n)
                                advanced_HSDP.gamma = fixed_gamma
                                advanced_HSDP.beta = fixed_beta
                                advanced_HSDP.k = k
                                advanced_HSDP.times = fixed_times
                                advanced_HSDP.sigma = fixed_sigma
                                # BBGN_recursive.times = fixed_times

                                # ================ Generate values and assign to advanced_HSDP ================
                                new_values = []
                                sorted_malicious = []
                                new_real_sums = []

                                malicious_users = []
                                if k > 0:
                                    malicious_users = set(random.sample(range(n), k))

                                for _ in range(fixed_times):
                                    new_values.append(value.copy())

                                    new_real_sums.append(sum(value))
                                    # ===================== Generate malicious_users =====================
                                    sorted_malicious.append(sorted(malicious_users))
                                advanced_HSDP.values = new_values
                                advanced_HSDP.real_sums = new_real_sums
                                advanced_HSDP.sorted_malicious = sorted_malicious
                                # BBGN_recursive.data = value

                                # Compare different protocols on the same values dataset
                                for protocol_name in protocols:
                                    # Create subfolder named after baseline in Result/
                                    baseline_folder = os.path.join(result_root,
                                                                   f"{protocol_name}",f"{distribution}")
                                    if not os.path.exists(baseline_folder):
                                        os.makedirs(baseline_folder)
                                    # problem_folder = os.path.join(baseline_folder, f"{problem}/eps")
                                    # if not os.path.exists(problem_folder):
                                    #     os.makedirs(problem_folder)
                                    # dataset_folder = os.path.join(problem_folder, "Simulated_data/eps")
                                    # if not os.path.exists(dataset_folder):
                                    #     os.makedirs(dataset_folder)
                                    # Define output filename

                                    outfile_name = f"result_n{n}_d{d - 1}_k{k}_n{n}_lam{lambda_n}.txt"
                                    outfile_path = os.path.join(baseline_folder, outfile_name)

                                    print("=========================")
                                    print(f"Running {protocol_name} with n={n}, domain={d} ...")
                                    start_time = time.time()

                                    handler = protocol_handlers.get(protocol_name)
                                    # if protocol_name == "BBGN_recursive":
                                    #     dp_sums, errors, nmessages_per_user = handler(BBGN_recursive.data)
                                    # else:
                                    dp_sums, errors, nmessages_per_user = handler(advanced_HSDP.values)

                                    end_time = time.time()
                                    cost_sec = end_time - start_time
                                    print(f"Done. Elapsed: {cost_sec:.4f} seconds.")
                                    print("=========================\n")

                                    # Write results to file
                                    with open(outfile_path, 'w', encoding='utf-8') as f:
                                        f.write(f"epsilon = {eps}" + "\n")
                                        f.write(
                                            f"Experiment with protocol = {protocol_name}, n={n}, domain={d - 1}\n")
                                        f.write(f"Elapsed time = {cost_sec:.4f} sec\n\n")
                                        f.write("lambda: " + str(advanced_HSDP.find_lambda(n)) + "\n")
                                        f.write("real sums: " + str(advanced_HSDP.real_sums) + "\n")
                                        f.write("DP sums: " + str(dp_sums) + "\n")
                                        f.write("Errors: " + str(errors) + "\n")
                                        f.write("Average error: " + str(np.mean(np.sort(errors)[10:-10])) + "\n")
                                        f.write("Avg RE: " + str(
                                            np.mean(np.sort(errors)[10:-10]) / advanced_HSDP.real_sums[0]) + "\n")
                                        f.write("#messages per user: " + str(nmessages_per_user) + "\n")
                                        f.write("Average #messages: " + str(np.mean(nmessages_per_user)) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
