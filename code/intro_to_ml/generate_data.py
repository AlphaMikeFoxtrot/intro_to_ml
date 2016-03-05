from datastore.models import *
from datastore import db
import random

def product(*args):
    return reduce(lambda x,y: x*y, args[0])

mapping = {
    1:sum,
    2:product
}

list_of_db_tables = []
for key in locals().keys():
    if "Data" in key:
        list_of_db_tables.append(locals()[key])

for Data in list_of_db_tables:
    num_params = 0
    for i in dir(Data):
        if "param" in i:
            num_params += 1
    for i in xrange(10000):
        params = [random.randint(0,200) for elem in xrange(num_params)]
        result = sum(params)
        data = params + [result]
        d = Data(*data)
        db.session.add(d)
        db.session.commit()
