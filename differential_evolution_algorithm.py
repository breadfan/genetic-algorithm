import numpy as np
class DifferentialEvolutionAlgorithm(list):
    def __init__(self, n_pop=10,
                 score_function=np.sum, crossover_prob=0.9, F=0.5,
                 bounds=(-1e-6, 1e6), n_epochs=5, minimize=False):
        """
        Initializes Differential Evolution algorithm. Create init population.
        Defines estimation function or const (F) for forward method of algorithm
        :param n_pop: size of population
        :param crossover_prob: float, probability to be crossovered, lies in [0, 1]
        :param F: const or func, main estimation function of the method. Used whilst mutation is performed
        :param bounds: bounds where lies solution
        :param score_function: optimized function
        :param n_epochs: number of iterations
        :param minimize: boolean, False by default
        """

        super().__init__()
        if not isinstance(bounds, np.ndarray):
            bounds = np.array(bounds)
        self.n_pop = n_pop
        self.n_dim = len(bounds)
        self.bounds = bounds
        self.pop = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0])*np.random.rand(n_pop, len(bounds))
        self.F_const = F   # real value, lies in [0, 2]
        # TODO: self.F_func = F_func or None
        self.crossover_prob = crossover_prob        # crossover probability value is high typically
        self.n_epochs = n_epochs
        self.minimize = minimize
        self.score_function = score_function
        self.scores = self.score_function(self.pop) # TODO: correct the function till we need to handle np.array

    def mutation(self, indices_3groups):
        """
        performs mutation by formula x_0 + F(x_1 - x_2)
        :param children: np.array of shape (n_pop, 3), with each values in each dimension in columns
                        mapped to a place in randomly selected trials.
        :return: mutated population, clipped to bounds if out of range.
        """
        first_group=self.pop[indices_3groups[:, 0]]
        second_group=self.pop[indices_3groups[:, 1]]
        third_group=self.pop[indices_3groups[:, 2]]

        mutated = np.clip(first_group+self.F_const*(second_group - third_group), self.bounds[:, 0], self.bounds[:, 1])

        return mutated
    def crossover(self, mutated):
        """
        performs crossover, which is mutation part de-facto.
        :param mutated: subset of population
        :return: changed population
        """
        probs = np.random.rand(mutated.shape[0], mutated.shape[1])
        val_ind = probs > self.crossover_prob

        mutated[val_ind] = self.pop[val_ind]
        return mutated

    def if_update_best(self, next, epoch):
        """
        Updates best value having an update.
        Prints new best element, ind of this element and its score.
        :param epoch: current epoch number
        :param next:
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
            print(f'Best element of population:\n{self.pop[best_ind]}\n and its score in decimal:'
                  f'\n{self.scores[best_ind]}')
            print('-----------------------------------------------------------------------------------')

    def choosing_trials(self):
        """
        Choosing three groups of elements for main DE algorithm: first group is simple range of the array, second and third — random indices.
        :return: np.ndarray of shape (self.n_pop, 3), with each column corresponding to each group of indices needed.
        """
        indices = range(self.n_pop)
        selected_indices = np.zeros((self.n_pop, 3), dtype=np.int32)

        for obj_id in range(self.n_pop-1):
            selected_indices[obj_id, 0] = obj_id

            selected_indices[obj_id, 1:] = np.random.choice \
                (np.concatenate((indices[:obj_id], indices[(obj_id+1):])), size=2, replace=False)
        selected_indices[-1, 0] = self.n_pop-1
        selected_indices[-1, 1:] = np.random.choice \
            (indices[:-1], size=2, replace=False)
        return selected_indices
    def simulate(self):
        best_ind, best_val = 0, self.scores[0].copy()
        self.best_value = best_val
        for ep in range(self.n_epochs):
            indices_3groups = self.choosing_trials()

            mutated = self.mutation(indices_3groups)
            next = self.crossover(mutated)
            self.if_update_best(next, ep)
            self.pop = next
            self.scores = self.score_function(self.pop)
