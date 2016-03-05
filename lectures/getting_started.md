#Machine Learning

#What is it?

Machine learning, loosely, is the descipline of computer science dedicated to having a computer create a mathematical model from data.  If correctly used, in conjunction with domain knowledge, a programmer needs little mathematical prowess to create mathematical models that perform as well as a mathematician with great abilities.  

More or less, I like to think of machine learning, as the automation of the mathematician.  There are those who will disagree with this definition, but I think it gets at the heart of the matter.  

For those of you who would disagree with me, feel free to read my defense below.  If however this is uninteresting to you, please feel free to skip ahead to the more interesting material.  

There are many among you who would see automation as a negative.  I disagree.  I'm not saying it's the best thing ever.  Or even always an even okay idea.  Jobs are lost.  People's lives are made worse (more often than not) and at the end of the day, once a job has been automated, it usually can't be unautomated.  However, when it comes to mathematics, I think automation is a good thing.  Because when can enjoy mathematics - the beauty of specific proofs still makes me smile.  However, mathematics isn't just beautiful, it's also supposed to be useful.  And unfortunately, most mathematics can be hard to work with - full of human error - and worse yet prone to errors that are not immediate while doing the calculation.  For this reason, it is hard to be a careful mathematician and a correct one.  Thus I believe computers are truly of use here.  

And I believe mathematics is more often done, because of computers.  There are certainly more machine learning experts out there than mathematicians.  And with market forces as they are, it's likely that number will continue to rise.  Therefore, I conclude that if mathematics and understanding of it is good, and that machine learning leads to more mathematics being done, then I can conclude that machine learning is good.  (Assuming a direct correlation).    Onward!

####DEFENSE OVER

So how does machine learning automate mathematics?  Mathematicians are living calculators.  They apply formulas to understand the world around us.  Unfortunately, knowing which formula to apply is trick and takes years of mastery.  With a machine learning expert, you can remove the application of formulas from the equation.  Only domain knowledge, an understanding of writing code (not an easy task mind you - but still easier than mathematics), and data are required to get your model right.  

##What you can do with the right model
Once you have a good model you can understand the world around you in, precise and exacting terms.  What does understanding give you?  The ability to predict the next president of the united states, the ability to shatter any object, by only interacting with it the right way, the ability travel in time (in theory), the ability to move an object at least as fast as the speed of light, the ability to teleport solid matter instanteanously (sort of in theory - this has been proven for small objects), the ability to regrow organs outside of the body, the ability to rewrite our dna for future generations, the ability to cure any disease, the ability to create machines capable of emotions.  

All of these things are in reach, if you understand the universe we live in well enough.  They require effort, and the right tools, but they are all more or less possible.  

##A first example

We'll talk about a lot of examples and go into specific techniques throughout.  But for now we are going to start with a simple example to motivate how easy, simple machine learning can be.  

Before we get started - I do have some assumptions about you - the reader.  

1) I assume you know python
2) I assume you understand some mathematics - statistics, calculus and linear algebra to be specific.  

If you don't know these things, don't worry you can learn them (I'll add resources here at some point).  Also, you can use google, coursera, edX, MIT OCW, khan academy.  But that's informal at best, right now.  

Our first example - teaching a machine to add 2 numbers.  
The full code:

```
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

```

This code may seem like a lot to take in, but it's actually pretty simple.  Most machine learning algorithms look the same as the above code - at least in the types of things you do.

First you generate training and test data:

`testing = Data2.query.limit(1000).all()`
`training = Data2.query.limit(60000).all()`
 
 You generate your model: 

 `constants = hill_climb(rebalance_strategy,training)`

 You validate your model:

 `percent_correct = validate(constants,testing)`

 Then you check your findings:

```
ave = mean([val[1] for val in possible_constants]) 
print "Mean:",ave
for const in possible_constants:
    print const
```

Now that we have a general sense of how to use machine learning algorithms, let's understand how to write them :)

```
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
```

All you need to do is initialize your model - in this case we only initialize our constants.  Then we apply our constants and the parameters to our guess at a functional form - our objective function.  

The objective function is the thing we are trying to optimize, to get our results to match up as close as possible to the dataset we have.  There is more or less discrete opitmization, which is basically just educated guessing.  That's right, almost all machine learning just boils down to guessing well.  That's where domain knowledge and some understanding of statistics comes in handy.  

We can actually think of our problem as a search problem - where we are looking for the right constants to make our objective function look like the results in our data.  In general this is called supervised learning and it's a whole lot easier than unsupervised learning (most of the time).  

When we are trying to simulate mathematics (and often even when we aren't), we can use fairly simple objective functions.  In this case we are making use of:

```
def objective_function(params,constants):
    return sum([params[ind]*val for ind,val in enumerate(constants)])
```

Which is just linear combinations of the parameters and the constants.  




