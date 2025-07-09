import math
import random
import numpy as np



class BBGN:
    def __init__(self, n, U, epsilon, delta):
        self.n = n  # the number of users
        self.U = U  # Domain size
        self.epsilon = epsilon
        self.delta = delta
        self.name = "BBGN"

        # what's this
        self.domain = n * U * 10  # Restrict all computations to finite field arithmetic
        self.sigma = 40  # statistical security parameter
        self.m = max(3,
                     int(math.ceil((2 * self.sigma + math.log2(self.domain)) / (math.log2(n) - math.log2(math.e)) + 1)))
        # m: number of messages per user sends. -> Higher user count enables fewer messages per user at equivalent security levels.

    def EstimateMessageNumber(self):
        return self.m

    def LocalRandomizer(self, value):
        # Make sure value is below U
        value = max(0, value)
        value = min(value, self.U)

        messages = []
        # add central noise
        value += np.random.negative_binomial(1.0 / self.n, 1 - math.exp(-self.epsilon / self.U))
        value -= np.random.negative_binomial(1.0 / self.n, 1 - math.exp(-self.epsilon / self.U))
        # random split to m shares
        # Increases privacy - higher shard count raises per-shard entropy.
        # Unsharded values are directly visible to analyzer.
        valuem = value
        for _ in range(self.m - 1):
            valuei = random.randint(0, self.domain - 1)
            valuem = (valuem + self.domain - valuei) % self.domain
            messages.append(valuei)
        messages.append(valuem)
        return messages

    def Analyzer(self, messages, values=''):
        dp_sum = sum(messages) % self.domain
        if dp_sum > self.domain / 2:
            dp_sum = dp_sum - self.domain
        if values == '':
            return dp_sum
        accurate_sum = sum(values)

        print("accurate sum = ", accurate_sum)
        print("dp sum = ", dp_sum)
        print("    |DP - ACC| = ", abs(dp_sum - accurate_sum))
        print("#messages/user = ", len(messages) / len(values))
        return dp_sum

    def Simulator(self, values):
        # Real values
        nmessages, dpsum = self.n * self.m, sum(values)

        # Central-DP values
        num_plus_1 = np.random.negative_binomial(1.0, 1 - math.exp(-self.epsilon / self.U))
        num_minus_1 = np.random.negative_binomial(1.0, 1 - math.exp(-self.epsilon / self.U))
        dpsum = dpsum + num_plus_1 - num_minus_1

        return nmessages, dpsum
