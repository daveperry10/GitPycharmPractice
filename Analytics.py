
import pandas as pd
import numpy as np

#Tested vs Scenario Matrix
def payoffDollars(initInvestmentPct, investorShare, origValue, newValue, discount):
    investment = (initInvestmentPct * origValue) * (1-discount)
    appreciationTotal = newValue - origValue + discount * origValue
    shareOfAppr = investorShare * appreciationTotal
    return max(investment + shareOfAppr,0)

def payoffIRR(initialInv, investorShare, apprAnnualized, life, discount):
    discountedInitialInv = initialInv * (1-discount)
    newHomeValue = ((1+apprAnnualized)**life)
    appreciationTotal = newHomeValue+discount-1
    shareOfAppr = investorShare * appreciationTotal
    # create sequence of zeros for IRR function
    # format: np.irr([cashOut, 0,0,0,0,0, cashIN])
    a = np.array([-discountedInitialInv])
    b = np.zeros(life - 1)
    c = np.array([discountedInitialInv + shareOfAppr])
    return np.irr(np.concatenate((a, b, c), axis=0))

#### HOME PRICE PATH ###
class HomePriceSimulation():
    """ class to create and use a path of home prices"""
    def __Init__(self, name, age):
        self.name = name
        self.age = age
        self.size = (10000,10)
        print("this is init function")
    def generateFromFile(self, filename):
        self.filename = filename
        print("this is generate function")




