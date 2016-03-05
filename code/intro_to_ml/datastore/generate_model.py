from sys import argv

number_of_possible_parameters = int(argv[1])

with open("models.py","w") as f:
    f.write("from datastore import db\n\n")
    for i in xrange(1,number_of_possible_parameters+1):
        f.write("class Data"+str(i)+" (db.Model):\n")
        f.write("\t__tablename__ = 'data"+str(i)+"'\n")
        f.write("\tid = db.Column(db.Integer, primary_key=True)\n")
        #creating the columns
        for elem in xrange(1,i+1):
            f.write("\tparam"+str(elem)+" = db.Column(db.Integer)\n")
        f.write("\tresult = db.Column(db.Integer)\n")
        #initializing the columns
        parameters = ["self"] + [ "param"+str(elem) for elem in xrange(1,i+1)] + ["result"]
        f.write("\n\tdef __init__("+",".join(parameters)+"):\n")
        for elem in parameters[1:]:
            f.write("\t\tself."+elem+" = "+elem+"\n")
        f.write("\n\n")
