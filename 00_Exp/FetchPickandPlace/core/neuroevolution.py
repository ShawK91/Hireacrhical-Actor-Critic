import random
import numpy as np
from scipy.special import expit
import fastrand, math


#Neuroevolution SSNE
class SSNE:
    def __init__(self, args):
        self.current_gen = 0
        self.args = args;
        self.population_size = self.args.pop_size;
        self.num_elitists = int(self.args.elite_fraction * args.pop_size)
        if self.num_elitists < 1: self.num_elitists = 1

    def selection_tournament(self, index_rank, num_offsprings, tournament_size):
        total_choices = len(index_rank)
        offsprings = []
        for i in range(num_offsprings):
            winner = np.min(np.random.randint(total_choices, size=tournament_size))
            offsprings.append(index_rank[winner])

        offsprings = list(set(offsprings))  # Find unique offsprings
        if len(offsprings) % 2 != 0:  # Number of offsprings should be even
            offsprings.append(offsprings[fastrand.pcg32bounded(len(offsprings))])
        return offsprings

    def list_argsort(self, seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    def regularize_weight(self, weight, mag):
        if weight > mag: weight = mag
        if weight < -mag: weight = -mag
        return weight

    def crossover_inplace(self, gene1, gene2):
        for param1, param2 in zip(gene1.parameters(), gene2.parameters()):

            # References to the variable tensors
            W1 = param1.data
            W2 = param2.data

            if len(W1.shape) == 2: #Weights no bias
                num_variables = W1.shape[0]
                # Crossover opertation [Indexed by row]
                num_cross_overs = fastrand.pcg32bounded(num_variables * 2)  # Lower bounded on full swaps
                for i in range(num_cross_overs):
                    receiver_choice = random.random()  # Choose which gene to receive the perturbation
                    if receiver_choice < 0.5:
                        ind_cr = fastrand.pcg32bounded(W1.shape[0])  #
                        W1[ind_cr, :] = W2[ind_cr, :]
                    else:
                        ind_cr = fastrand.pcg32bounded(W1.shape[0])  #
                        W2[ind_cr, :] = W1[ind_cr, :]

            elif len(W1.shape) == 1: #Bias
                num_variables = W1.shape[0]
                # Crossover opertation [Indexed by row]
                num_cross_overs = fastrand.pcg32bounded(num_variables)  # Lower bounded on full swaps
                for i in range(num_cross_overs):
                    receiver_choice = random.random()  # Choose which gene to receive the perturbation
                    if receiver_choice < 0.5:
                        ind_cr = fastrand.pcg32bounded(W1.shape[0])  #
                        W1[ind_cr] = W2[ind_cr]
                    else:
                        ind_cr = fastrand.pcg32bounded(W1.shape[0])  #
                        W2[ind_cr] = W1[ind_cr]

    def mutate_inplace(self, gene):
        mut_strength = 0.05
        num_mutation_frac = 0.1
        super_mut_strength = 10
        super_mut_prob = 0.05
        reset_prob = super_mut_prob + 0.05

        num_params = len(list(gene.parameters()))
        ssne_probabilities = np.random.uniform(0, 1, num_params) * 2

        for i, param in enumerate(gene.parameters()): #Mutate each param

            # References to the variable keys
            W = param.data
            if len(W.shape) == 2: #Weights, no bias

                num_weights= W.shape[0]*W.shape[1]
                ssne_prob = ssne_probabilities[i]

                if random.random()<ssne_prob:
                    num_mutations = fastrand.pcg32bounded(int(math.ceil(num_mutation_frac * num_weights)))  # Number of mutation instances
                    for _ in range(num_mutations):
                        ind_dim1 = fastrand.pcg32bounded(W.shape[0])
                        ind_dim2 = fastrand.pcg32bounded(W.shape[-1])
                        random_num = random.random()

                        if random_num < super_mut_prob:  # Super Mutation probability
                            W[ind_dim1, ind_dim2] += random.gauss(0, super_mut_strength * W[ind_dim1, ind_dim2])
                        elif random_num < reset_prob:  # Reset probability
                            W[ind_dim1, ind_dim2] = random.gauss(0, 1)
                        else:  # mutauion even normal
                            W[ind_dim1, ind_dim2] += random.gauss(0, mut_strength *W[ind_dim1, ind_dim2])

                        # Regularization hard limit
                        W[ind_dim1, ind_dim2] = self.regularize_weight(W[ind_dim1, ind_dim2], self.args.weight_magnitude_limit)

                elif len(W.shape) == 1: #Bias
                    num_weights = W.shape[0]
                    ssne_prob = ssne_probabilities[i]

                    if random.random() < ssne_prob:
                        num_mutations = fastrand.pcg32bounded(
                            int(math.ceil(num_mutation_frac * num_weights)))  # Number of mutation instances
                        for _ in range(num_mutations):
                            ind_dim = fastrand.pcg32bounded(W.shape[0])
                            random_num = random.random()

                            if random_num < super_mut_prob:  # Super Mutation probability
                                W[ind_dim] += random.gauss(0, super_mut_strength * W[ind_dim])
                            elif random_num < reset_prob:  # Reset probability
                                W[ind_dim] = random.gauss(0, 1)
                            else:  # mutauion even normal
                                W[ind_dim] += random.gauss(0, mut_strength * W[ind_dim])

                            # Regularization hard limit
                            W[ind_dim] = self.regularize_weight(W[ind_dim], self.args.weight_magnitude_limit)

    def tmutate_inplace(self, hive):
        mut_strength = 0.05
        num_mutation_frac = 0.1
        super_mut_strength = 10
        super_mut_prob = 0.05
        reset_prob = super_mut_prob + 0.05

        for drone in hive.all_drones:
            # References to the variable keys
            keys = list(drone.param_dict.keys())
            W = drone.param_dict
            num_structures = len(keys)
            ssne_probabilities = np.random.uniform(0,1,num_structures)*2

            for ssne_prob, key in zip(ssne_probabilities, keys): #For each structure
                if random.random()<ssne_prob:

                    mut_matrix = scipy_rand(W[key].shape[0], W[key].shape[1], density=num_mutation_frac, data_rvs=np.random.randn).A * mut_strength
                    W[key] += np.multiply(mut_matrix, W[key])


                    # num_mutations = fastrand.pcg32bounded(int(math.ceil(num_mutation_frac * W[key].size)))  # Number of mutation instances
                    # for _ in range(num_mutations):
                    #     ind_dim1 = fastrand.pcg32bounded(W[key].shape[0])
                    #     ind_dim2 = fastrand.pcg32bounded(W[key].shape[-1])
                    #     random_num = random.random()
                    #
                    #     if random_num < super_mut_prob:  # Super Mutation probability
                    #         W[key][ind_dim1, ind_dim2] += random.gauss(0, super_mut_strength *
                    #                                                                       W[key][
                    #                                                                           ind_dim1, ind_dim2])
                    #     elif random_num < reset_prob:  # Reset probability
                    #         W[key][ind_dim1, ind_dim2] = random.gauss(0, 1)
                    #
                    #     else:  # mutauion even normal
                    #         W[key][ind_dim1, ind_dim2] += random.gauss(0, mut_strength *W[key][
                    #                                                                           ind_dim1, ind_dim2])
                    #
                    #     # Regularization hard limit
                    #     W[key][ind_dim1, ind_dim2] = self.regularize_weight(
                    #         W[key][ind_dim1, ind_dim2])

    def clone(self, master, replacee):  # Replace the replacee individual with master
        for target_param, source_param in zip(replacee.parameters(), master.parameters()):
            target_param.data.copy_(source_param.data)

    def reset_genome(self, gene):
        for param in (gene.parameters()):
            param.data.copy_(param.data)

    def epoch(self, pop, fitness_evals):

        # Entire epoch is handled with indices; Index rank nets by fitness evaluation (0 is the best after reversing)
        index_rank = self.list_argsort(fitness_evals); index_rank.reverse()
        elitist_index = index_rank[:self.num_elitists]  # Elitist indexes safeguard

        # Selection step
        offsprings = self.selection_tournament(index_rank, num_offsprings=len(index_rank) - self.num_elitists,
                                               tournament_size=3)

        # #Extinction step (Resets all the offsprings genes; preserves the elitists)
        # if random.random() < self.args.extinction_prob: #An extinction event
        #     print()
        #     print("######################Extinction Event Triggered#######################")
        #     print()
        #     for i in offsprings:
        #         if random.random() < self.args.extinction_magnituide and not (i in elitist_index):  # Extinction probabilities
        #             self.reset_genome(pop[i])

        # Figure out unselected candidates
        unselects = []; new_elitists = []
        for i in range(self.population_size):
            if i in offsprings or i in elitist_index:
                continue
            else:
                unselects.append(i)
        random.shuffle(unselects)

        # Elitism step, assigning elite candidates to some unselects
        for i in elitist_index:
            replacee = unselects.pop(0)
            new_elitists.append(replacee)
            self.clone(master=pop[i], replacee=pop[replacee])

        # Crossover for unselected genes with 100 percent probability
        if len(unselects) % 2 != 0:  # Number of unselects left should be even
            unselects.append(unselects[fastrand.pcg32bounded(len(unselects))])
        for i, j in zip(unselects[0::2], unselects[1::2]):
            off_i = random.choice(new_elitists);
            off_j = random.choice(offsprings)
            self.clone(master=pop[off_i], replacee=pop[i])
            self.clone(master=pop[off_j], replacee=pop[j])
            self.crossover_inplace(pop[i], pop[j])

        # Crossover for selected offsprings
        for i, j in zip(offsprings[0::2], offsprings[1::2]):
            if random.random() < self.args.crossover_prob: self.crossover_inplace(pop[i], pop[j])

        # Mutate all genes in the population except the new elitists
        for i in range(self.population_size):
            if i not in new_elitists:  # Spare the new elitists
                if random.random() < self.args.mutation_prob: self.mutate_inplace(pop[i])

        #Pick Worst index
        while True:
            worst_index = fastrand.pcg32bounded(len(pop)-1)
            if worst_index not in new_elitists: break

        return new_elitists[0], worst_index



