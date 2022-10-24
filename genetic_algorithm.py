import numpy as np
class GeneticAlgorithm(list):
    def __init__(self, n_pop=10, n_bits=10, select_k=3,
                 score_function=np.sum, crossover_prob=0.9, bounds=None,
                 n_epochs=5, mut_prob=None, minimize=False):
        super().__init__()
        self.n_pop = n_pop
        self.n_dim = len(bounds)
        self.n_bits = n_bits*len(bounds)  # multiplying because of groups
        self.pop = np.random.randint(0, 2, (n_pop, n_bits*len(bounds)))
        self.bounds = bounds  # for continuous function
        self.select_k = select_k

        self.mut_prob = mut_prob or 1.0 / n_bits   # mutation probability value is low typically
        self.crossover_prob = crossover_prob        # crossover probability value is high typically
        self.n_epochs = n_epochs
        self.minimize = minimize
        self.decoded = None
        self.score_function = score_function
        if score_function != np.sum:
            self.decoded = self.decode()
            self.scores = score_function(self.decoded)
        else:
            self.scores = score_function(self.pop, axis=1) # 2-d array
    def decode(self):
        """
            Taking a binary array of arrays (or bitstrings),
            function returns array of decoded to decimal numeric system array of arrays of numers.
            Vectorized.
            :return decoded: array of arrays with decimal numbers.
        """
        decoded = np.zeros((self.n_pop, self.n_dim))

        for group_num in np.arange(self.n_dim):
            begin, end = group_num*self.n_bits//self.n_dim, (group_num+1)*self.n_bits//self.n_dim
            temp_slice = self.pop[:, begin:end]

            temp_slice = temp_slice.dot(1 << np.arange(temp_slice.shape[-1] - 1, -1, -1))
            # putting values into an interval [a, b]
            # formulae is: a_i + k*(b_i - a_i)/2^n, where a, b -- borders, n -- number of bits
            denom = 2**(self.n_bits//self.n_dim)   # assuming n_bits for each group
            a, b = self.bounds[group_num]

            temp_slice = a + temp_slice*(b - a)/denom

            decoded[np.arange(self.n_pop), group_num] =  temp_slice
        return decoded

    def selection(self):
        """
        Searching for the best candidate for continuing population.
        Operation is vectorized.

        :return pop[best_cand_ids]: the best possible objects for continuing the population
        """
        best_cand_ids = np.random.randint(0, self.n_pop, self.n_pop)
        rand_cand_ids = np.random.randint(0, self.n_pop, (self.n_pop, self.select_k-1))

        if self.minimize:
            rand_cand_ids = rand_cand_ids[np.arange(self.n_pop), np.argmin(self.scores[rand_cand_ids],axis=1)] # for minimization
        else:
            rand_cand_ids = rand_cand_ids[np.arange(self.n_pop), np.argmax(self.scores[rand_cand_ids],axis=1)]  # watching at k random candidates

        # selecting only one for each gene

        mult = -1 if self.minimize else 1  # if we minimize the score function, simply multiply by 1
        ids_where_best_worse = mult*self.scores[rand_cand_ids] > mult*self.scores[best_cand_ids]
        best_cand_ids[ids_where_best_worse] = rand_cand_ids[ids_where_best_worse]


        parents = self.pop[best_cand_ids]
        # if len(parents) == 1:
        #     parents= np.vstack([parents, parents])

        return parents
    def mutation(self, children):
        """
            Performs mutation over children.
            Vectorized.
            Rewrites self.pop
            :param : current row to implement mutation
        """
        probs = np.random.rand(children.shape[0], self.n_bits)  # probs of mutations

        children[probs > self.mut_prob] = 1 - children[probs > self.mut_prob]

        return children

    def crossover(self, parents):
        pairs_ids = np.stack([np.arange(0, self.n_pop-1, 2), np.arange(1, self.n_pop, 2)], axis=1)  # just matrix (n-1, 2) with pairs of indexes: [[0, 1], [2, 3], [4, 5]...]

        probs_of_selection = np.random.rand(self.n_pop//2)  # probs of selection
        valid_pairs = pairs_ids[probs_of_selection <= self.crossover_prob]  # "<" because we're draw probability of "not to be crossovered"
        not_valid_pairs = pairs_ids[probs_of_selection > self.crossover_prob]
        change_ptr = np.random.randint(1, self.n_bits-2, valid_pairs.shape[0])

        children = parents[not_valid_pairs].tolist()

        for el_ind in np.arange(len(valid_pairs)): # no way of vectorization since pointers may vary

            first, second = parents[valid_pairs[el_ind]]
            res_first, res_second = first.copy(), second.copy()
            # changing substrings

            res_first[:change_ptr[el_ind]] = second[:change_ptr[el_ind]] # '+' operation works as an .append() function
            res_second[:change_ptr[el_ind]] = first[:change_ptr[el_ind]]

            #assert type(res_first) == np.ndarray
            children.append([res_first, res_second])
        children = np.vstack(children)

        return children
    def if_update_best(self, epoch):
        """
        Updates best value if have an update.
        Prints
        :param epoch: current epoch number

        """
        update_best = False
        if self.minimize:
            best_ind_temp = np.argmin(self.scores)
            if self.scores[best_ind_temp] < self.best_value:
                update_best = True
        else:
            best_ind_temp = np.argmax(self.scores)
            if self.scores[best_ind_temp] > self.best_value:
                update_best = True
        if update_best:
            best_ind = best_ind_temp.copy()
            best_val = self.scores[best_ind].copy()
            self.best_value = best_val
            print(f'Epoch {epoch}. The best element now is: {best_ind} with value: {best_val}.')
            print(f'Best element itself in binary code:\n{self.pop[best_ind]}\n and decode in decimal:'
                  f'\n{self.decoded[best_ind]}')
            print('-----------------------------------------------------------------------------------')

    def simulate(self):
        best_ind, best_val = 0, self.scores[0].copy()
        self.best_value = best_val
        for epoch in range(self.n_epochs):
            self.if_update_best(epoch)

            # main part of the algorithm
            parents = self.selection()
            children = self.crossover(parents)
            mutate = self.mutation(children)

            self.pop = mutate.copy()
            if self.score_function == np.sum:
                self.scores = self.score_function(self.pop, axis=1)
            else:
                self.decoded = self.decode()
                self.scores = self.score_function(self.decoded)


            if best_val == self.n_bits and self.score_function == np.sum:
                print(f'Found max with on {epoch} epoch')
                return