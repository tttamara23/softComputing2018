from numpy import exp, random, dot

class Neuronska():
    def __init__(self):
        # generisanje istih brojeva svaki put kad se program pokrene
        random.seed(1)

        #svaki neuron ima tri ulaza(y koordinate one tri tacke) i jedan izlaz - koji kaze 0 ili 1
        #0 tuzan, 1 srecan
        #inicijalizujemo tezine na random brojeve
        self.tezine = 2 * random.random((5, 5)) - 1

    #sigmoidna funkcija za aktivaciju neurona
    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def obuci(self, ulazi, izlazi, iteracije):

        for i in xrange(iteracije):
            predvidjeniIzlaz = self.predvidi(ulazi)
           #racuna gresku kao razliku izmedju zeljenog izlaza i predvidjenog izlaza
            greska = izlazi - predvidjeniIzlaz

            #poboljsavanje tezina, tj navodimo mrezu da da izlaz blizi zeljenom
            #propagacija unazad
            prilagodjavanje = dot(ulazi.T, greska * self.sigmoid_derivative(predvidjeniIzlaz))
            self.tezine += prilagodjavanje

    #predvidjanje neuronske mreze
    def predvidi(self, ulazi):
        return self.sigmoid(dot(ulazi, self.tezine))
