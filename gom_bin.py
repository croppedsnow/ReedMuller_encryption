import itertools
import numpy as np
from scipy.special import comb
from time import perf_counter

class ReedMuller:
    def __init__(self, r, m):
        self.__r = r
        self.__m = m
        self.__n = 2 ** m
        self.__k = int(np.sum(comb(np.full((1, r + 1), m), np.arange(0, r + 1))))
        self.__d = 2 ** (m - r)
        self.__G = self.gen_g()
        self.__sk = self.keygen()

    def gen_g(self):
        G = np.zeros((self.__k, self.__n), dtype=np.bool_)
        G[0, :] = 1

        G[1:self.__m + 1, :] = ((np.arange(2 ** self.__m, dtype=int)[:, np.newaxis] >> np.arange(self.__m - 1, -1, -1, dtype=int)) & 1).T.astype(np.bool_)

        pos = self.__m + 1

        for i in range(2, self.__r + 1):
            combin = np.array(list(itertools.combinations(list(range(1, self.__m + 1)), i)), dtype=int)
            cur_count = combin.shape[0]
            G[pos:pos + cur_count, :] = np.prod(G[combin], axis=1) % 2
            pos += cur_count

        return G

    def keygen(self):
        P = np.random.permutation(self.__k * self.__n).reshape(self.__k, self.__n)
        P = np.stack((np.zeros((self.__k, self.__n), dtype=int), P), axis=2, dtype=int)
        P[:, :, 0] = P[:, :, 1] // self.__n
        P[:, :, 1] = P[:, :, 1] % self.__n

        S = np.random.choice(range(0, self.__n), size=np.random.randint(self.__d / 2 + 1, self.__d), replace=False)
        return (S, P)

    def encrypt(self, mes):
        W = mes[:, np.newaxis] * self.__G
        E = np.zeros(W.shape, dtype=np.bool_)
        E[:, self.__sk[0]] = np.random.randint(0, 2, size=(self.__k, self.__sk[0].shape[0]), dtype=np.bool_)
        W = W ^ E
        W1 = np.zeros((self.__k, self.__n), dtype=np.bool_)
        for i in range(self.__k):
            for j in range(self.__n):
                W1[self.__sk[1][i, j][0], self.__sk[1][i, j][1]] = W[i, j]
        return W1

    def decrypt(self, C):
        W = np.zeros((self.__k, self.__n), dtype=np.bool_)
        for i in range(self.__k):
            for j in range(self.__n):
                W[i, j] = C[self.__sk[1][i, j][0], self.__sk[1][i, j][1]]

        W = np.bitwise_xor.reduce(W, axis=0)
        return self.major_decode(W, self.__sk[0])

    def major_decode(self, W, S):
        mes = np.zeros(self.__k, dtype=np.bool_)
        pos = self.__k
        for i in range(self.__r, 0, -1):
            combin = np.array(list(itertools.combinations(list(range(self.__m)), i)), dtype=int)
            pos -= combin.shape[0]
            table = self.__G[i + 1:self.__m + 1, (self.__n // 2 ** i):].T
            for t in range(combin.shape[0]):
                cur_ind = 0
                for j in range(2 ** (self.__m - i)):
                    cur_ind = np.where(np.prod(self.__G[np.delete(np.arange(1, self.__m + 1, dtype=int), combin[t, :]), :] == table[j].reshape(self.__m - i, 1), axis=0))[0]
                    if not np.intersect1d(S, cur_ind).shape[0]: ######## ПОМЕНЯЯЯЯЯТЬ
                        break
                mes[pos + t] = np.bitwise_xor.reduce(W[cur_ind])

            W = W ^ np.bitwise_xor.reduce(mes[pos : pos + combin.shape[0], np.newaxis] * self.__G[pos : pos + combin.shape[0], :], axis=0)

        i = 0
        while i in S:
            i += 1
        mes[0:1] = W[i]

        return mes

    def get_k(self):
        return self.__k

    def get_G(self):
        return self.__G

    def get_sk(self):
        return self.__sk

t_sum = 0
for i in range(100):
    for m in range(11, 12):
        for r in range(2, 3):
            t_begin = perf_counter()
            coder = ReedMuller(r, m)
            mes = np.random.randint(0, 2, size = coder.get_k(), dtype=np.bool_)

            cip = coder.encrypt(mes)

            dec = coder.decrypt(cip)

            res = np.prod(mes == dec)
            t_end = perf_counter() - t_begin
            t_sum += t_end
            print(r, m, res, t_end)
            if not res:
                print("Error")
                exit(1)
print("All right")
print(t_sum / 100)