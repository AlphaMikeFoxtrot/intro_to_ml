import numpy as np

def check_index(cur_index,size):
    if size - abs(cur_index) > 0:
        return "keep going"
    else:
        return "stop"

def rate_of_change(a,b):
    try:
        return float(b-a)/2
    except:
        return 0 
        
def rates_of_change(listing):
    listing2 = listing[1:]
    return map(rate_of_change,listing,listing2)[:-1]

def check_sign(num):
    if num >= 0:
        return "positive"
    else:
        return "negative"
    
def find_inflection_points(listing):
    rates = rates_of_change(listing)
    inflection_points = []
    num_inflection_points = 0
    sign = check_sign(rates[0]) 
    for index,rate in enumerate(rates):
        new_sign = check_sign(rate)
        if new_sign != sign:
            inflection_points.append(rate)
            num_inflection_points += 1
            sign = new_sign
    return inflection_points,num_inflection_points

def get_hidden_layer_index(nn):
    indexes = []
    for index,syn in enumerate(nn):
        if type(syn["name"]) == type(int()):
            indexes.append(index)
    return indexes

def sort_by_key(listing,sort_by):
    """
    Expects a list of dictionaries 
    Returns a list of dictionaries sorted by the associated key
    """
    keys = []
    translate = {}
    for ind,elem in enumerate(listing):
        key = elem[sort_by]
        translate[key] = ind
        keys.append(key)
    keys.sort()
    new_ordering = [translate[key] for key in keys]
    return [listing[i] for i in new_ordering]
