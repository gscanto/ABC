from statistics import mean
from sklearn.model_selection import cross_val_score
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import sys
import argparse
import numpy as np
from abc import ABCMeta
from six import add_metaclass
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm


@add_metaclass(ABCMeta)
class ArtificialBee(object): # Abelhas   
    TRIAL_INITIAL_DEFAULT_VALUE = 0
    INITIAL_DEFAULT_PROBABILITY = 0.0

    def __init__(self, source_food,fitness):
        self.source_food = source_food
        self.fitness = fitness
        self.trial = ArtificialBee.TRIAL_INITIAL_DEFAULT_VALUE
        self.prob = ArtificialBee.INITIAL_DEFAULT_PROBABILITY

    def compute_prob(self, max_fitness):
        self.prob = self.fitness / max_fitness
    
    def get_fitness(self):
        return 1 / (1 + self.fitness) if self.fitness >= 0 else 1 + np.abs(self.fitness)


    def explore(self, food, MR):
        # parâmetro de pertubação
        for i in range(len(food)):
            R = np.random.uniform(low=0, high=1)
            if R < MR:
                food[i] = 1
            else:
                continue

        return food

    def random_solutions(source_food, n_features):
    
        i = np.random.randint(0, n_features)
        k = np.random.randint(0, n_features)
        # XOR
        if np.random.uniform(low=0, high=1) < 0.5:
            gamma = np.bool_(1)
        else:
            gamma = np.bool_(0)
        source_food[i] = np.int64(np.bool_(source_food[i]) ^ gamma ^ np.bool_(source_food[i]) ^ np.bool_(source_food[k]))
        return source_food


class ABC(object):

    def __init__(self, X, y, estimator, scorer, n_iter, max_trials, MR, k_folds):
        self.X = X
        self.y = y
        self.estimator = estimator
        self.scorer = scorer
        self.n_features = len(X[0])

        self.n_iter = n_iter
        self.max_trials = max_trials
        self.MR = MR
        self.k_folds = k_folds

        self.optimal_solution = None

        self.bees = []

    def initialize(self):   
        initial_population = np.zeros((self.n_features,self.n_features), dtype=np.int64)
        
        # cria populacao inicial
        for itr in range(0, self.n_features):
            initial_population[itr][itr] = 1

            # Calcula score populacao inicial
            self.X_selected = self.X[:, np.array(initial_population[itr,:], dtype=np.bool)]

            # Pontua a populacao inicial
            scores = cross_val_score(estimator=self.estimator, X= self.X_selected, y= self.y, scoring= self.scorer,cv= self.k_folds)
            scores_mean = np.mean(scores)

            self.bees.append(ArtificialBee(initial_population[itr], scores_mean))
     

    def employee_bees_phase(self):
        for itr in self.bees:
            
            search_neighbor = itr.explore(itr.source_food, self.MR)
            
            # calcula score do vizinho
            scores_neighbor = cross_val_score(estimator=self.estimator, X= self.X[:, np.array(search_neighbor, dtype=np.bool)], y= self.y, 
                                        scoring= self.scorer,cv= self.k_folds)
            scores_mean_neighbor = np.mean(scores_neighbor)
            # verifica se o vizinho tem uma pontuacao melhor que a solucao atual
            
            if (itr.fitness < scores_mean_neighbor):
                self.bees.append(ArtificialBee(search_neighbor, scores_mean_neighbor))
            else:
                itr.trial =+ 1
     

    def __onlooker_bees_phase(self):
        for itr in self.best_food_sources:

            search_neighbor = itr.explore(itr.source_food, self.MR)
            
            # calcula score do vizinho
            scores_neighbor = cross_val_score(estimator=self.estimator, X= self.X[:, np.array(search_neighbor, dtype=np.bool)], y= self.y, 
                                        scoring= self.scorer,cv= self.k_folds)
            scores_mean_neighbor = np.mean(scores_neighbor)
            # verifica se o vizinho tem uma pontuacao melhor que a solucao atual
            
            if (itr.fitness < scores_mean_neighbor):
                self.bees.append(ArtificialBee(search_neighbor, scores_mean_neighbor))
            else:
                itr.trial =+ 1

        self.best_food_sources = []

    def __scout_bee_phase(self):
        for itr in self.bees:
            if (itr.trial > self.max_trials):
                itr.source_food = itr.random_solutions(itr.source_food, self.n_features)
                score = cross_val_score(estimator=self.estimator, X= self.X[:, np.array(itr.source_food, dtype=np.bool)], y= self.y, 
                                        scoring= self.scorer,cv= self.k_folds)
                itr.fitness = np.mean(score)
                
    def __update_optimal_solution(self):
        n_optimal_solution = \
            max(self.bees, key=lambda bee: bee.fitness)
        if not self.optimal_solution:
            self.optimal_solution = deepcopy(n_optimal_solution)
        else:
            if n_optimal_solution.fitness < self.optimal_solution.fitness:
                self.optimal_solution = deepcopy(n_optimal_solution)
    
    def __calculate_probabilities(self):
        sum_fitness = sum(map(lambda bee: bee.get_fitness(), self.bees))
    
        for itr in self.bees:
            itr.compute_prob(sum_fitness)

    def __select_best_food_sources(self):
        self.best_food_sources = list(
            filter(lambda bee: bee.prob > np.random.uniform(low=0, high=1), self.bees))
        while not self.best_food_sources:
            self.best_food_sources = list( 
            filter(lambda bee: bee.prob > np.random.uniform(low=0, high=1), self.bees))
    
    def feature_selection(self):
        self.initialize()
        for itr in range(self.n_iter):
            self.employee_bees_phase()
            self.__calculate_probabilities()
            self.__select_best_food_sources()
            self.__onlooker_bees_phase()
            self.__scout_bee_phase()    
            self.__update_optimal_solution()
        return self.optimal_solution.source_food

def float_range(mini,maxi):
    # Define the function with default arguments
    def float_range_checker(arg):
        try:
            f = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError("Must be a Floating Point Number")
        if f <= mini or f >= maxi:
            raise argparse.ArgumentTypeError("Must be > " + str(mini) + " and < " + str(maxi))
        return f
    return float_range_checker

def parse_args(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-d', '--dataset', metavar='DATASET',
        help='Dataset (csv file).', type=str, required=True)
    parser.add_argument(
        '-e', '--estimator', metavar='ESTIMATOR',
        help="Estimator.",
        choices=['svm', 'rf', 'knn', 'dt'],
        type=str, default='svm')
    parser.add_argument(
        '-es', '--escorer', metavar='ESCORER',
        help="Escorer.",
        choices=['accuracy', 'f1-score', 'recall', 'precision'],
        type=str, default='accuracy')
    parser.add_argument(
        '-it', '--iteration', metavar='ITERATION',
        help="ITERATION.",
        type=int, default=10)
    parser.add_argument(
        '-max', '--max-trials', metavar='MAX-trials',
        help="Max-trials.",
        type=int, default=10)
    parser.add_argument(
        '--mr', metavar='MR',
        help="MUTATION.",
        type=float_range(0.0, 1.0), default=0.5)
    parser.add_argument(
        '-k', '--k-fold', metavar='K',
        help="K.",
        type=int,
        default=3)
    args = parser.parse_args(argv)
    return args

def get_X_y(dataset):
    X = dataset.drop(columns = 'class')
    y = dataset['class']
    return X, y

def create_csv(dataset, selected_columns, name):
    new_columns = []
    for itr in range(len(selected_columns)):
        if selected_columns[itr] == 1:
            new_columns.append(dataset.columns[itr])
    
    new_columns.append(dataset.columns[-1])
    dataset[new_columns].to_csv("selected_" + name)
        
if __name__=="__main__":
    args = parse_args(sys.argv[1:])

    try:
        dataset = pd.read_csv(args.dataset)
        #dataset['class'] = dataset['class'].map({'benign': 0,
        #                            'malign': 1})
    except BaseException as e:
        print('Exception: {}'.format(e))
        exit(1)
    

    if args.estimator == 'svm':
        clf = svm.SVC()
    if args.estimator == 'rf':
        clf = RandomForestClassifier(random_state = 0)
    if args.estimator == 'knn':
        clf = KNeighborsClassifier()
    if args.estimator == 'dt':
        clf = DecisionTreeClassifier()

    
    
    X, y = get_X_y(dataset)
    
    X_np = X.to_numpy()
    y_np = y.to_numpy()
    abc = ABC(X_np, y_np, clf, args.escorer, args.iteration, args.max_trials, args.mr, args.k_fold)
    features_selected = abc.feature_selection()
    create_csv(dataset, features_selected, args.dataset)
