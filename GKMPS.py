import math
import numpy as np
import random


class GKMPS:
    def __init__(self, n, domain, epsilon, delta, gamma):
        self.n = n  # the number of users
        self.domain = domain  # Domain size
        self.name = "GKMPS"  # name for hsdp
        scale = 2  # change the scale factor for different bucket sizes
        if domain > math.ceil(epsilon * math.sqrt(n) * scale):
            self.U = math.ceil(epsilon * math.sqrt(n) * scale)  # try a small domain
            self.B = math.ceil(domain / self.U)  # bucket size
        else:
            self.U = domain
            self.B = 1
        self.epsilon = epsilon
        self.delta = delta
        self.epsilonstar = (1 - gamma) * epsilon
        self.epsilon1 = min(1.0, gamma * epsilon) * 0.5
        self.epsilon2 = min(1.0, gamma * epsilon) * 0.5
        self.delta1 = delta / 2
        self.delta2 = delta / 2

        self.t = np.zeros(2 * self.U + 1)  # self.t is an array of size [-U, U]
        Gamma = self.U * (math.ceil(np.log(self.U)) + 1)
        for i in range(-self.U, self.U + 1, 1):
            if i != 0:
                self.t[i] = math.ceil(Gamma / abs(i))

    def RandomizedRounding(self, x):
        # Make sure value is below U
        x = max(0, x)
        x = min(x, self.domain)

        x = x / self.B
        prob = x - math.floor(x)
        if random.random() <= prob:
            x = math.ceil(x)
        else:
            x = math.floor(x)
        return x

    ###############################################################################################################
    #
    #   LocalRandomizer 
    #   Input: value (the user hold) 
    #   Output: a list of messages the user sent to secure shuffler
    #
    ###############################################################################################################
    def LocalRandomizer(self, value):
        # real mesasge
        messages = []
        if value != 0:
            messages.append(self.RandomizedRounding(value))

        # central noise
        num_plus_1 = np.random.negative_binomial(1.0 / self.n, 1 - math.exp(-self.epsilonstar / self.U))
        messages += [1] * num_plus_1
        num_minus_1 = np.random.negative_binomial(1.0 / self.n, 1 - math.exp(-self.epsilonstar / self.U))
        messages += [-1] * num_minus_1

        # sum zero noises
        z = np.zeros(2 * self.U + 1)

        for i in range(-self.U, self.U + 1, 1):
            if i == 0 or i == -1:
                continue
            if i == 1:
                z[i] += np.random.negative_binomial(3 * (1 + np.log(1 / self.delta1)) / self.n,
                                                    1 - math.exp(-0.2 * self.epsilon1 / self.U))
            z[i] += np.random.negative_binomial(3 * (1 + np.log((2 * self.U - 1) / self.delta2)) / self.n,
                                                1 - math.exp(-0.1 * self.epsilon2 / self.t[i]))

        for i in range(-self.U, self.U + 1, 1):
            if i == 0 or i == -1:
                continue
            elif i == 1:
                messages += [-1, 1] * int(z[i])
            else:
                messages += [i, (- i // 2), (- i - (-i // 2))] * int(z[i])

        return messages

    ###############################################################################################################
    #
    #   Analyzer 
    #   Input: a list of messages from all users (ignore shuffling)
    #   Output: sum of the messages
    #
    ###############################################################################################################
    def Analyzer(self, messages, values=''):
        if values == '':
            return sum(messages) * self.B

        # Debug
        rrvalues = []
        for value in values:
            rrvalues.append(self.RandomizedRounding(value))

        accurate_sum = sum(values)
        rr_sum = sum(rrvalues) * self.B
        dp_sum = sum(messages) * self.B
        print("accurate sum = ", accurate_sum)
        print("random rounding sum = ", rr_sum)
        print("dp sum = ", dp_sum)
        print("    |DP - ACC| = ", abs(dp_sum - accurate_sum))
        print("    |DP - RR| = ", abs(dp_sum - rr_sum))
        print("    |RR - ACC| = ", abs(rr_sum - accurate_sum))
        print("#messages/user = ", len(messages) / len(values))

        return dp_sum

    ###############################################################################################################
    #
    #   EstimateMessageNumber 
    #   Input: value (the user hold) 
    #   Output: the number of messages the user sent (expected)
    #
    ###############################################################################################################
    def EstimateMessageNumber(self, value=''):
        message = 0

        # need to send real mesasge
        if value != 0 and value != '':
            message += 1

        # central noise
        message += 2 * (1.0 / self.n * self.U / self.epsilonstar)

        # sum zero noises
        for i in range(-self.U, self.U + 1, 1):
            if i == 0 or i == -1:
                continue
            if i == 1:
                message += 2 * (3 * (1 + np.log(1 / self.delta1)) / self.n * self.U / (0.2 * self.epsilon1))
            message += 3 * (
                    3 * (1 + np.log((2 * self.U - 1) / self.delta2)) / self.n * self.t[i] / (0.1 * self.epsilon2))
        return message

    ###############################################################################################################
    #
    #   Simulator 
    #   Input: a list of input values
    #   Output: the total number of received messages, the DP noised sum (in practice)
    #
    ###############################################################################################################
    def Simulator(self, values):
        # Real values
        nmessages, dpsum = 0, 0
        for value in values:
            rvalue = self.RandomizedRounding(value)
            if (rvalue != 0):
                nmessages += 1
            dpsum += rvalue

        # Central-DP values
        print(1 - math.exp(-self.epsilonstar / self.U))
        print(self.epsilonstar)
        print(self.U)
        num_plus_1 = np.random.negative_binomial(1.0, 1 - math.exp(-self.epsilonstar / self.U))
        num_minus_1 = np.random.negative_binomial(1.0, 1 - math.exp(-self.epsilonstar / self.U))
        nmessages += num_plus_1 + num_minus_1
        dpsum = dpsum + num_plus_1 - num_minus_1

        # Zero-sum DP values
        z = np.zeros(2 * self.U + 1)
        for i in range(-self.U, self.U + 1, 1):
            if i == 0 or i == -1:
                continue
            if i == 1:
                z[i] += np.random.negative_binomial(3 * (1 + np.log(1 / self.delta1)),
                                                    1 - math.exp(-0.2 * self.epsilon1 / self.U))
            z[i] += np.random.negative_binomial(3 * (1 + np.log((2 * self.U - 1) / self.delta2)),
                                                1 - math.exp(-0.1 * self.epsilon2 / self.t[i]))

        for i in range(-self.U, self.U + 1, 1):
            if i == 0 or i == -1:
                continue
            elif i == 1:
                nmessages += 2 * int(z[i])
            else:
                nmessages += 3 * int(z[i])

        return nmessages, dpsum * self.B

    ###############################################################################################################
    #
    #   Simulator for HSDP
    #   Difference: doubles noise
    #
    ###############################################################################################################
    def Simulator_for_HSDP(self, values, honest_user_proportion):
        # Real values
        nmessages, dpsum = 0, 0
        for value in values:
            rvalue = self.RandomizedRounding(value)
            if (rvalue != 0):
                nmessages += 1
            dpsum += rvalue

        # Central-DP values
        num_plus_1 = np.random.negative_binomial(2.0 * honest_user_proportion, 1 - math.exp(-self.epsilonstar / self.U))
        num_minus_1 = np.random.negative_binomial(2.0 * honest_user_proportion,
                                                  1 - math.exp(-self.epsilonstar / self.U))

        nmessages += num_plus_1 + num_minus_1
        dpsum = dpsum + num_plus_1 - num_minus_1

        # Zero-sum DP values
        z = np.zeros(2 * self.U + 1)
        for i in range(-self.U, self.U + 1, 1):
            if i == 0 or i == -1:
                continue
            if i == 1:
                z[i] += np.random.negative_binomial(6 * honest_user_proportion * (1 + np.log(1 / self.delta1)),
                                                    1 - math.exp(-0.2 * self.epsilon1 / self.U))
            z[i] += np.random.negative_binomial(
                6 * honest_user_proportion * (1 + np.log((2 * self.U - 1) / self.delta2)),
                1 - math.exp(-0.1 * self.epsilon2 / self.t[i]))

        for i in range(-self.U, self.U + 1, 1):
            if i == 0 or i == -1:
                continue
            elif i == 1:
                nmessages += 2 * int(z[i])
            else:
                nmessages += 3 * int(z[i])

        return nmessages, dpsum * self.B

    ###############################################################################################################
    #
    #   Simulator for GKMPS when k>0
    #   Difference: doubles noise
    #
    ###############################################################################################################
    def Simulator_for_GKMPS_k(self, values, honest_user_proportion):
        # Real values
        nmessages, dpsum = 0, 0
        for value in values:
            rvalue = self.RandomizedRounding(value)
            if (rvalue != 0):
                nmessages += 1
            dpsum += rvalue
        # print(f"U: {self.U}")
        # print(f"Bits per message: {1 + math.ceil(math.log2(self.U + 1))}")
        # Central-DP values
        num_plus_1 = np.random.negative_binomial(honest_user_proportion, 1 - math.exp(-self.epsilonstar / self.U))
        num_minus_1 = np.random.negative_binomial(honest_user_proportion,
                                                  1 - math.exp(-self.epsilonstar / self.U))
        nmessages += num_plus_1 + num_minus_1
        # print("central", (num_plus_1 + num_minus_1)/len(values))
        dpsum = dpsum + num_plus_1 - num_minus_1
        # Zero-sum DP values
        z = np.zeros(2 * self.U + 1)
        for i in range(-self.U, self.U + 1, 1):
            if i == 0 or i == -1:
                continue
            if i == 1:
                z[i] += np.random.negative_binomial(3 * honest_user_proportion * (1 + np.log(1 / self.delta1)),
                                                    1 - math.exp(-0.2 * self.epsilon1 / self.U))
                # print(f"delta:{self.delta1},count1:{z[i]/len(values)}")

            count = np.random.negative_binomial(
                3 * honest_user_proportion * (1 + np.log((2 * self.U - 1) / self.delta2)),
                1 - math.exp(-0.1 * self.epsilon2 / self.t[i]))
            z[i] += count
            # print(f"count2:{count / len(values)}")
        for i in range(-self.U, self.U + 1, 1):
            if i == 0 or i == -1:
                continue
            elif i == 1:
                nmessages += 2 * int(z[i])

            else:
                nmessages += 3 * int(z[i])

        return nmessages, dpsum * self.B


n = 512
values = [random.choice(range(2)) for _ in range(n)]
print(values[1:10])
gkmps = GKMPS(n, 2, 0.0333, 1 / n / n, 0.3)
nmessage, dp_sum = gkmps.Simulator(values)
# print(dp_sum)
# print(sum(values))
# print(dp_sum - sum(values))
print(nmessage)
# print(1 - math.exp(0.7 / 1))
# for _ in range(20):
#     print(np.random.negative_binomial(0.9, 1 - math.exp(-0.7 / 2)))
#     print(np.random.negative_binomial(1, 1 - math.exp(-0.7 / 2)))
#     print("---")
