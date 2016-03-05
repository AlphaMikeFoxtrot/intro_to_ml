from datastore.models import *
import random

def objective_function(constants,parameters):
    return sum([c*p for c,p in zip(constants,parameters)])

def score_function(first,second):
    return abs(first - second)

def strat(constants,scores,epsilon=0.05):
    cur_score = scores[-1]
    prev_scores = scores[:-1]
    mean_score = sum(prev_scores)/float(len(prev_scores))
    total = []
    if cur_score < epsilon: return constants 
    if cur_score < mean_score:
        if random.randint(0,1000) %2 ==0:
            for ind,elem in enumerate(constants):
                if ind%2==0:
                    total.append(elem-0.01)
                else:
                    total.append(elem+0.01)
            return total
        else:
            for ind,elem in enumerate(constants):
                if ind%2==0:
                    total.append(elem+0.01)
                else:
                    total.append(elem-0.01)
            return total
    else:
        if random.randint(0,1000) %2 ==0:
            for ind,elem in enumerate(constants):
                if ind%2==0:
                    total.append(elem-0.1)
                else:
                    total.append(elem+0.1)
            return total
        else:
            for ind,elem in enumerate(constants):
                if ind%2==0:
                    total.append(elem+0.1)
                else:
                    total.append(elem-0.1)
            return total
        
def hill_climb(strategy,data,constants):
    scores = [0]
    for i in data:
        parameters = [i.__dict__[j] for j in i.__dict__.keys() if "param" in j]
        result = objective_function(constants,parameters)
        scores.append(score_function(result,i.result))
        #xsprint scores[-1]
        constants = strategy(constants,scores)
    return constants,scores


def tuner(data,num_params):
    best = 1000
    initializers = []
    initializers.append([0 for i in xrange(num_params)])
    initializers.append([1 for i in xrange(num_params)])
    initializers.append([-1 for i in xrange(num_params)])
    initializers.append([2 for i in xrange(num_params)])
    initializers.append([-2 for i in xrange(num_params)])

    for i in xrange(10):
        initializers.append([random.randint(0,1)+random.random() for i in xrange(num_params)])
        initializers.append([random.randint(-2,2)+random.random() for i in xrange(num_params)])
    
    final_constants = []
    final_scores = []
    for ind,initial in enumerate(initializers):
        constants,temp_scores = hill_climb(strat,data,initial)
        if sum(temp_scores)/float(len(temp_scores)) < best:
            final_constants = constants
            final_scores = temp_scores
            best = sum(temp_scores)/float(len(temp_scores))
            print best
    return final_constants,final_scores
    
def main():
    d_3 = Data3.query.limit(10000).all()
    constants,scores = tuner(d_3,3) 
    scores.sort()
    print "constants were:",constants
    print "average score",sum(scores)/float(len(scores))
    print "ten best scores",scores[:10]

if __name__ == '__main__':
   main()
    
