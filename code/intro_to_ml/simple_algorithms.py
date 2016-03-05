from datastore.models import *
import random
import math

def product(listing):
    return reduce(lambda x,y:x*y,listing)

def objective_function(params,constants):
    return sum([params[ind]*val for ind,val in enumerate(constants)])

def score_function(anticipated_result,actual_result):
    return abs(anticipated_result-actual_result) #lower score is better (means the result is closer to what you actually got)

def rebalance(constants,amount):
    if random.randint(0,100) % 2 == 0:
        constants = [constants[0]-amount,constants[1]+amount]
    else:
        constants = [constants[0]+amount,constants[1]-amount]
    return constants

def rebalance_strategy(scores,threshold,constants):
    score = scores[-1]
    prev_score = scores[-2]
    if score > prev_score and score > threshold:
        constants = rebalance(constants,1/float(score))
    else:
        constants = rebalance(constants,0.01)
    return constants

def get_params(datum):
    params = []
    dicter = datum.__dict__
    for key in dicter:
        if "param" in key:
            params.append(dicter[key])
    return params

def hill_climb(rebalance_strategy,training):
    constants = [0,0]
    threshold = 10
    scores = [1000]
    for elem in training:
        params = get_params(elem)
        result = objective_function(params,constants)
        score = score_function(result,elem.result)
        scores.append(score)
        constants = rebalance_strategy(scores,threshold,constants)
    return constants

def validate(constants, test_data):
    total = len(test_data)
    correct = 0
    for i in test_data:
        params = get_params(i)
        expected_result = objective_function(params,constants)
        if expected_result - i.result < 0.001:
            correct += 1
    return float(correct)/total

def mean(vals):
    return sum(vals)/float(len(vals))

if __name__ == '__main__':

    possible_constants = []
    for i in xrange(100):
        testing = Data2.query.limit(1000).all()
        training = Data2.query.limit(60000).all()
    
        constants = hill_climb(rebalance_strategy,training)
        percent_correct = validate(constants,testing)
        possible_constants.append((constants,percent_correct))

    ave = mean([val[1] for val in possible_constants]) 
    print "Mean:",ave
    for const in possible_constants:
        print const
