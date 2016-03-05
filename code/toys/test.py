from datastore.models import *

for key in locals().keys():
    if "Data" in key:
        print dir(locals()[key])
        break
