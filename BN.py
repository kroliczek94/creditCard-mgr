from pomegranate import BayesianNetwork
import numpy
import seaborn, time
def run():

    seaborn.set_style('whitegrid')

    X = numpy.random.randint(2, size=(2000, 7))
    X[:, 3] = X[:, 1]
    X[:, 6] = X[:, 1]

    X[:, 0] = X[:, 2]

    X[:, 4] = X[:, 5]

    model = BayesianNetwork.from_samples(X, algorithm='exact')

    model.structure
    model.plot()

run()