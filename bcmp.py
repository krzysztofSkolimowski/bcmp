import json
import numpy as np
from numpy import linalg, math
from enum import Enum


# Słowniczek:
# Dane wejściowe:
# n[i] - typ i-tego węzła (systemu kolejkowyego)
# u[i] - intensywność obsługi zgłoszeń i-tego węzła (dla każdej klasy zgłoszenia)
# m[i] - liczba kanałów obsługi i-tego węzła
# p[r] - macierz prawdopodobieństwa przejść między węzłami dla klasy zgłoszeń r
# k[r] - liczba zgłoszeń dla r-tej klas
# Dane wynikowe:
# K[i][r] - średnia liczba zgłoszeń klasy r w systemie i
# T[i][r] - średni czas przebywania zgłoszenia klasy r w systemie i
# W[i][r] - średni czas oczekiwania zgłoszeń klasy r w systemie i
# Q[i][r] - średnia długość kolejki zgłoszeń klasy r w systemie i

# Numery wzorków wzięte z http://home.agh.edu.pl/~kwiecien/ : 
# Modele Kolejkowe w IT : 
# pdf: Materiały przydatne do projektu z sieci BCMP (metoda SUM) 

# Loads network model data from json file
def load_data():
    with open('model.json') as f:
        model = json.load(f)
    return model


class ModelState(Enum):
    INITIAL = 0
    INVALID = 1
    VALID = 2


class GNetwork:
    def __init__(self, model):
        self.model = model
        self.N = len(model["n"])
        self.R = len(model["k"])
        self.compute_network_properties()

    def compute_network_properties(self):
        self.e = self.compute_visit_ratios(round_to=0.001)
        self.lm = self.model["epsilon"] * np.ones(self.R)  # lambdas[r]
        self.lm2 = np.zeros((self.N, self.R))  # lambdas[i][r]
        self.ro = np.zeros(self.N)  # ro[i]
        self.ro2 = np.zeros((self.N, self.R))  # ro[i][r]
        self.kSum = sum(self.model["k"])
        self.mState = ModelState.INITIAL
        self.adjustment_iteration = 0
        self.apply_sum_method(iterationLimit=1000)

    # Computes formula 4.23 
    # for network with class switching not allowed
    # assumed e1r = 1
    # return e: size ixr
    def compute_visit_ratios(self, round_to):
        e = np.zeros((self.N, self.R))
        for r in range(0, self.R):
            A = np.zeros((self.N, self.N))
            for i in range(0, self.N):
                for j in range(0, self.N):
                    A[j][i] = self.model["p"][r][i][j]
                    if i == j:
                        A[i][j] -= 1
            b = -A[:, 0]
            A = np.delete(A, 0, 1)
            out, _, _, _ = linalg.lstsq(A, b, rcond=round_to)
            for i in range(0, self.N):
                if i == 0:
                    e[i][r] = 1
                else:
                    e[i][r] = out[i - 1]
        return e

    # Computes formula 4.16
    def compute_ro(self, i, r):
        self.lm2[i][r] = self.lm[r] * self.e[i][r]
        if self.model["n"][i] == 1:
            self.ro2[i][r] = self.lm2[i][r] / (self.model["m"][i] * self.model["u"][i][r])
        else:
            self.ro2[i][r] = self.lm2[i][r] / self.model["u"][i][r]

    # Computes formula 4.54 (below)
    def compute_pmi(self, i):
        coefficient_a = (self.model["m"][i] * self.ro[i]) ** self.model["m"][i]
        coefficient_b = (1 - self.ro[i]) * math.factorial(self.model["m"][i])
        coefficient_c, coefficient_d, coefficient_e, coefficient_f = [0] * 4
        for k in range(0, self.model["m"][i] - 1):
            coefficient_d = ((self.model["m"][i] * self.ro[i]) ** k) / math.factorial(k)
            coefficient_e = ((self.model["m"][i] * self.ro[i]) ** self.model["m"][i]) / math.factorial(
                self.model["m"][i])
            coefficient_f = 1 / (1 - self.ro[i])
            coefficient_c += coefficient_d + coefficient_e * coefficient_f
        return (coefficient_a / coefficient_b) * (1 / coefficient_c)

    # Computes first formula in fix 4.55
    def get_fix_formula_a(self, i, r):
        numerator = self.e[i][r] / self.model["u"][i][r]
        denominator = 1 - ((self.kSum - 1) / self.kSum) * self.ro[i]
        return numerator / denominator

    # Computes second formula in fix 4.55
    def getFixFormulaB(self, i, r):
        coefficientA = self.e[i][r] / self.model["u"][i][r]
        numerator = self.e[i][r] / (self.model["u"][i][r] * self.model["m"][i])
        denominator = 1 - ((self.kSum - self.model["m"][i] - 1) / (self.kSum - self.model["m"][i])) * self.ro[i]
        coefficientB = self.compute_pmi(i)
        return coefficientA + (numerator / denominator) * coefficientB

    # Computes third formula in fix 4.55
    def getFixFormulaC(self, i, r):
        return self.e[i][r] / self.model["u"][i][r]

    # Computes fix 4.55
    def computeFix(self, i, r):
        fix = 0.0
        if self.model["m"][i] == 1 and self.model["n"][i] in set([1, 2]):  # type 4 not allowed
            fix = self.get_fix_formula_a(i, r)
        elif self.model["m"][i] > 1 and self.model["n"][i] == 1:
            fix = self.getFixFormulaB(i, r)
        elif self.model["n"][i] == 3:
            fix = self.getFixFormulaC(i, r)
        else:
            raise Exception('Invalid n or m values')
        return fix

    # Computes one iteration of 4.55
    def iterations_strategy(self):
        for r in range(0, self.R):
            fix_sum = 0
            for i in range(0, self.N):
                ro = 0
                for r2 in range(0, self.R):
                    self.compute_ro(i, r2)
                    ro += self.ro2[i][r2]
                self.ro[i] = ro
                fix = self.computeFix(i, r)
                fix_sum += fix
            if fix_sum == 0:
                raise Exception('division by zero')
            self.lm[r] = self.model["k"][r] / fix_sum

    # Determines if ro values are correct
    def check_ro_values(self):
        for r in range(0, self.R):
            for i in range(0, self.N):
                if self.model["n"][i] != 3 and self.ro2[i][r] > 1:
                    np.set_printoptions(precision=2)
                    np.set_printoptions(suppress=True)
                    print('Ro values: ')
                    print(self.ro2)
                    if self.adjustment_iteration > 1000:
                        print('\nUnable to adjust number of service channels\n')
                        raise Exception('Incorrect values of ro for system nr ' + str(i))
                    return i
        return -1

    # Uses SUM Method to approximate lambda values
    def apply_sum_method(self, iterationLimit):
        i = 0
        while True:
            lm_prev = np.copy(self.lm)
            self.iterations_strategy()
            error = 0
            for r in range(0, self.R):
                error += (lm_prev[r] - self.lm[r]) ** 2
            error = math.sqrt(error)
            if error < self.model["epsilon"] or i > iterationLimit:
                break
            i += 1
        check_ro = self.check_ro_values()
        if check_ro != -1:
            self.mState = ModelState.INVALID
            self.model["m"][check_ro] += 1
            self.adjustment_iteration += 1
            self.apply_sum_method(iterationLimit)
        else:
            if self.mState == ModelState.INVALID:
                self.mState = ModelState.VALID
                print('\nNumber of service channels adjusted: ')
                print(self.model["m"])
                print('')

    # Computes first formula in 4.54
    def get_k_formula_a(self, i, r):
        return self.ro2[i][r] / (1 - ((self.kSum - 1) / self.kSum) * self.ro[i])

    # Computes second formula in 4.54
    def get_k_formula_b(self, i, r):
        coefficientA = self.model["m"][i] * self.ro2[i][r]
        coefficientB = 1 - ((self.kSum - self.model["m"][i] - 1) / (self.kSum - self.model["m"][i])) * self.ro[i]
        return coefficientA + (self.ro2[i][r] / coefficientB) * self.compute_pmi(i)

    # Computes third formula in 4.54
    def get_k_formula_c(self, i, r):
        return self.lm2[i][r] / self.model["u"][i][r]

    # Computes whole formula 4.54
    def compute_average_k(self):
        K = np.zeros((self.N, self.R))
        for r in range(0, self.R):
            for i in range(0, self.N):
                if self.model["m"][i] == 1 and self.model["n"][i] in set([1, 2]):  # type 4 not allowed
                    K[i][r] = self.get_k_formula_a(i, r)
                elif self.model["m"][i] > 1 and self.model["n"][i] == 1:
                    K[i][r] = self.get_k_formula_b(i, r)
                elif self.model["n"][i] == 3:
                    K[i][r] = self.get_k_formula_c(i, r)
                else:
                    raise Exception('Invalid n or m values')
        return K

    # Computes formula 4.26 
    def compute_average_t(self, K):
        T = np.zeros((self.N, self.R))
        for r in range(0, self.R):
            for i in range(0, self.N):
                if self.lm2[i][r] == 0:
                    T[i][r] = float('nan')
                else:
                    T[i][r] = K[i][r] / self.lm2[i][r]
        return T

    # Computes formula 4.27 
    def compute_average_w(self, T):
        W = np.zeros((self.N, self.R))
        for r in range(0, self.R):
            for i in range(0, self.N):
                W[i][r] = T[i][r] - 1 / self.model["u"][i][r]
        return W

    # Computes formula 4.25
    def compute_average_q(self, W):
        Q = np.zeros((self.N, self.R))
        for r in range(0, self.R):
            for i in range(0, self.N):
                Q[i][r] = self.lm2[i][r] * W[i][r]
        return Q


def main():
    model = load_data()
    q_n = GNetwork(model)
    K = q_n.compute_average_k()
    T = q_n.compute_average_t(K)
    W = q_n.compute_average_w(T)
    Q = q_n.compute_average_q(W)
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)
    print('Row: Queueing system number')
    print('Column: Request class')
    print('K: ')
    print(K)
    print('T: ')
    print(T)
    print('W: ')
    print(W)
    print('Q: ')
    print(Q)


if __name__ == '__main__':
    main()
