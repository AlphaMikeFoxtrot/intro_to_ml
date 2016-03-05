from datastore.models import *
import random
import math
import scipy as sp
from scipy.stats import mstats
from scipy import stats

def product(listing):
    return reduce(lambda x,y:x*y,listing)

def objective_function(params,constants):
    poly = [[] for elem in constants]
    for ind,const in enumerate(constants):
        return poly[ind].append(sum([math.pow(const,i)*params[ind] for i in xrange(1,10)]))

def score_function(anticipated_result,actual_result):
    return abs(anticipated_result-actual_result) #lower score is better (means the result is closer to what you actually got)


#adjust terms less than
def adjust_terms_lt(const,modifier,index):
    if index % 2 == 0:
        return const - modifier
    else:
        return const + modifier
    
#adjust terms greater than
def adjust_terms_gt(const,modifier,index):
    if index % 2 == 0:
        return const + modifier
    else:
        return const - modifier

def rebalance_strategy1(scores,episolon,constants):
    score = scores[-1]
    prev_score = scores[-2]
    if score == 0: return score
    if abs(score - prev_score) < episolon:
        modifier = 1/float(score*score)
        if prev_score < score:
            constants = [adjust_terms_lt(const,modifier,ind) for ind,const in enumerate(constants)]
        if prev_score > score:
            constants = [adjust_terms_gt(const,modifier,ind) for ind,const in enumerate(constants)]
    else:
        modifier = 1/(score*0.5)
        if prev_score < score:
            constants = [adjust_terms_lt(const,modifier,ind) for ind,const in enumerate(constants)]
        if prev_score > score:
            constants = [adjust_terms_gt(const,modifier,ind) for ind,const in enumerate(constants)]
    return constants

def rebalance_strategy2(scores,episolon,constants):
    mean_score = sum(scores)/float(len(scores))
    cur_score = scores[-1]
    if mean_score <= episolon:
        modifier = 1/(1.0 + cur_score)
        if cur_score < mean_score:
            constants = [adjust_terms_lt(const,modifier,ind) for ind,const in enumerate(constants)]
        elif abs(cur_score - mean_score) < 1.0:
            constants = [adjust_terms_gt(const,modifier,ind) for ind,const in enumerate(constants)]
        else:
            modifier = 1/float(cur_score)
            constants = [adjust_terms_gt(const,modifier,ind) for ind,const in enumerate(constants)]
    else:
        if cur_score < mean_score:
            modifier = 1/float(mean_score)
            constants = [adjust_terms_lt(const,modifier,ind) for ind,const in enumerate(constants)]
        else:
            modifier = 1/float(math.log(mean_score))
            constants = [adjust_terms_gt(const,modifier,ind) for ind,const in enumerate(constants)]
    return constants

def first_four_moments(arr):
    arr = sp.array(arr)
    _,_,mean,variance,skew,kurtosis = stats.describe(arr)
    return [mean,variance,skew,kurtosis]

def get_moments(scores):
    return first_four_moments(scores) + [mstats.moment(scores,moment=i) for i in xrange(5,11)]

def rate_of_change(a,b):
    return float(b-a)/2

def describe_dynamics(arr):
    rates_of_change = [rate_of_change(arr[ind-1],elem) for ind,elem in enumerate(arr[1:])]
    ave_rate_of_change = sum(rates_of_change)/float(len(rates_of_change))
    return rates_of_change[-1],ave_rate_of_change #returns the most recent rate of change and the average rate of change

def split_by_index(arrs):
    new_arrs = [[] for elem in xrange(len(arrs[0]))]
    for arr in arrs:
        for ind,val in enumerate(arr):
            new_arrs[ind].append(val)
    return new_arrs

#consider dampening for higher order terms
def rebalance_strategy3(scores,episolon,constants,total_moments=[]):
    #we need two sets of moments to calculate rates of change
    if len(total_moments) < 2: return rebalance_strategy2(scores,episolon,constants)
    else:
        moments = split_by_index(total_moments)
        dynamics = [describe_dynamics(moment) for moment in moments[:3]]
        for dynamic in dynamics:
            cur_score = scores[-1]
            mean_score = sum(scores)/float(len(scores))
            modifier = 1/float(mean_score)
            if dynamic[0] > 0 and dynamic[1] > 0:
                constants = [adjust_terms_lt(const,modifier,ind) for ind,const in enumerate(constants)]
            if dynamic[0] > 0 and dynamic[1] <= 0:
                modifier = 1/float(math.log(mean_score))
                constants = [adjust_terms_lt(const,modifier,ind) for ind,const in enumerate(constants)]
            else:
                constants = [adjust_terms_gt(const,modifier,ind) for ind,const in enumerate(constants)]
        return constants
#next strategy - allow for code to vary based on cur_score and how much better I've done the last 10 or 20 times (a quickly fading smoothing affect)

def gen_start(x):
    if x%2==0: return 1
    return 0

def get_params(db_obj):
    dicter = db_obj.__dict__
    return [dicter[key] for key in dicter if "param" in key]       

def hill_climb(rebalance_strategy,training,num_const,account_for_moments=False):
    constants = [gen_start(i) for i in xrange(num_const)]  
    episolon = 0.0005
    scores = [100000]
    count = 0
    moments = []
    for elem in training:
        params = get_params(elem)
        result = objective_function(params,constants)
        score = score_function(result,elem.result)
        if score == 0:
            print "score function completely fits data!!!"
            break
        scores.append(score)
        if account_for_moments:
            moments.append(get_moments(scores))
            constants = rebalance_strategy(scores,episolon,constants,total_moments=moments)
        else:
            constants = rebalance_strategy(scores,episolon,constants)
    return scores 

if __name__ == '__main__':
    dbs = []
    for key in locals().keys():
        if "Data" in key:
            num_consts = int(key.split("Data")[1])
            dbs.append([num_consts,locals()[key]])
            
    for num_constants,db in dbs:
        training = db.query.limit(5000).all()
        print "Case",num_constants
        scores1 = hill_climb(rebalance_strategy1,training,num_constants)
        scores2 = hill_climb(rebalance_strategy2,training,num_constants)
        #scores3 = hill_climb(rebalance_strategy3,training,num_constants,account_for_moments=True)
        scores1.sort()
        scores2.sort()
        #scores3.sort()
        print "strategy 1",sum(scores1)/float(len(scores1))
        print scores1[:10]
        print "strategy 2",sum(scores2)/float(len(scores2))
        print scores2[:10]
        #print "strategy 3",sum(scores3)/float(len(scores3))
        #print scores3[:10]
        
        #print count/float(1000)
        #print constants
