from datastore import db

class Data1 (db.Model):
	__tablename__ = 'data1'
	id = db.Column(db.Integer, primary_key=True)
	param1 = db.Column(db.Integer)
	result = db.Column(db.Integer)

	def __init__(self,param1,result):
		self.param1 = param1
		self.result = result


class Data2 (db.Model):
	__tablename__ = 'data2'
	id = db.Column(db.Integer, primary_key=True)
	param1 = db.Column(db.Integer)
	param2 = db.Column(db.Integer)
	result = db.Column(db.Integer)

	def __init__(self,param1,param2,result):
		self.param1 = param1
		self.param2 = param2
		self.result = result


class Data3 (db.Model):
	__tablename__ = 'data3'
	id = db.Column(db.Integer, primary_key=True)
	param1 = db.Column(db.Integer)
	param2 = db.Column(db.Integer)
	param3 = db.Column(db.Integer)
	result = db.Column(db.Integer)

	def __init__(self,param1,param2,param3,result):
		self.param1 = param1
		self.param2 = param2
		self.param3 = param3
		self.result = result


class Data4 (db.Model):
	__tablename__ = 'data4'
	id = db.Column(db.Integer, primary_key=True)
	param1 = db.Column(db.Integer)
	param2 = db.Column(db.Integer)
	param3 = db.Column(db.Integer)
	param4 = db.Column(db.Integer)
	result = db.Column(db.Integer)

	def __init__(self,param1,param2,param3,param4,result):
		self.param1 = param1
		self.param2 = param2
		self.param3 = param3
		self.param4 = param4
		self.result = result


class Data5 (db.Model):
	__tablename__ = 'data5'
	id = db.Column(db.Integer, primary_key=True)
	param1 = db.Column(db.Integer)
	param2 = db.Column(db.Integer)
	param3 = db.Column(db.Integer)
	param4 = db.Column(db.Integer)
	param5 = db.Column(db.Integer)
	result = db.Column(db.Integer)

	def __init__(self,param1,param2,param3,param4,param5,result):
		self.param1 = param1
		self.param2 = param2
		self.param3 = param3
		self.param4 = param4
		self.param5 = param5
		self.result = result


class Data6 (db.Model):
	__tablename__ = 'data6'
	id = db.Column(db.Integer, primary_key=True)
	param1 = db.Column(db.Integer)
	param2 = db.Column(db.Integer)
	param3 = db.Column(db.Integer)
	param4 = db.Column(db.Integer)
	param5 = db.Column(db.Integer)
	param6 = db.Column(db.Integer)
	result = db.Column(db.Integer)

	def __init__(self,param1,param2,param3,param4,param5,param6,result):
		self.param1 = param1
		self.param2 = param2
		self.param3 = param3
		self.param4 = param4
		self.param5 = param5
		self.param6 = param6
		self.result = result


class Data7 (db.Model):
	__tablename__ = 'data7'
	id = db.Column(db.Integer, primary_key=True)
	param1 = db.Column(db.Integer)
	param2 = db.Column(db.Integer)
	param3 = db.Column(db.Integer)
	param4 = db.Column(db.Integer)
	param5 = db.Column(db.Integer)
	param6 = db.Column(db.Integer)
	param7 = db.Column(db.Integer)
	result = db.Column(db.Integer)

	def __init__(self,param1,param2,param3,param4,param5,param6,param7,result):
		self.param1 = param1
		self.param2 = param2
		self.param3 = param3
		self.param4 = param4
		self.param5 = param5
		self.param6 = param6
		self.param7 = param7
		self.result = result


class Data8 (db.Model):
	__tablename__ = 'data8'
	id = db.Column(db.Integer, primary_key=True)
	param1 = db.Column(db.Integer)
	param2 = db.Column(db.Integer)
	param3 = db.Column(db.Integer)
	param4 = db.Column(db.Integer)
	param5 = db.Column(db.Integer)
	param6 = db.Column(db.Integer)
	param7 = db.Column(db.Integer)
	param8 = db.Column(db.Integer)
	result = db.Column(db.Integer)

	def __init__(self,param1,param2,param3,param4,param5,param6,param7,param8,result):
		self.param1 = param1
		self.param2 = param2
		self.param3 = param3
		self.param4 = param4
		self.param5 = param5
		self.param6 = param6
		self.param7 = param7
		self.param8 = param8
		self.result = result


class Data9 (db.Model):
	__tablename__ = 'data9'
	id = db.Column(db.Integer, primary_key=True)
	param1 = db.Column(db.Integer)
	param2 = db.Column(db.Integer)
	param3 = db.Column(db.Integer)
	param4 = db.Column(db.Integer)
	param5 = db.Column(db.Integer)
	param6 = db.Column(db.Integer)
	param7 = db.Column(db.Integer)
	param8 = db.Column(db.Integer)
	param9 = db.Column(db.Integer)
	result = db.Column(db.Integer)

	def __init__(self,param1,param2,param3,param4,param5,param6,param7,param8,param9,result):
		self.param1 = param1
		self.param2 = param2
		self.param3 = param3
		self.param4 = param4
		self.param5 = param5
		self.param6 = param6
		self.param7 = param7
		self.param8 = param8
		self.param9 = param9
		self.result = result


class Data10 (db.Model):
	__tablename__ = 'data10'
	id = db.Column(db.Integer, primary_key=True)
	param1 = db.Column(db.Integer)
	param2 = db.Column(db.Integer)
	param3 = db.Column(db.Integer)
	param4 = db.Column(db.Integer)
	param5 = db.Column(db.Integer)
	param6 = db.Column(db.Integer)
	param7 = db.Column(db.Integer)
	param8 = db.Column(db.Integer)
	param9 = db.Column(db.Integer)
	param10 = db.Column(db.Integer)
	result = db.Column(db.Integer)

	def __init__(self,param1,param2,param3,param4,param5,param6,param7,param8,param9,param10,result):
		self.param1 = param1
		self.param2 = param2
		self.param3 = param3
		self.param4 = param4
		self.param5 = param5
		self.param6 = param6
		self.param7 = param7
		self.param8 = param8
		self.param9 = param9
		self.param10 = param10
		self.result = result


class Data11 (db.Model):
	__tablename__ = 'data11'
	id = db.Column(db.Integer, primary_key=True)
	param1 = db.Column(db.Integer)
	param2 = db.Column(db.Integer)
	param3 = db.Column(db.Integer)
	param4 = db.Column(db.Integer)
	param5 = db.Column(db.Integer)
	param6 = db.Column(db.Integer)
	param7 = db.Column(db.Integer)
	param8 = db.Column(db.Integer)
	param9 = db.Column(db.Integer)
	param10 = db.Column(db.Integer)
	param11 = db.Column(db.Integer)
	result = db.Column(db.Integer)

	def __init__(self,param1,param2,param3,param4,param5,param6,param7,param8,param9,param10,param11,result):
		self.param1 = param1
		self.param2 = param2
		self.param3 = param3
		self.param4 = param4
		self.param5 = param5
		self.param6 = param6
		self.param7 = param7
		self.param8 = param8
		self.param9 = param9
		self.param10 = param10
		self.param11 = param11
		self.result = result


class Data12 (db.Model):
	__tablename__ = 'data12'
	id = db.Column(db.Integer, primary_key=True)
	param1 = db.Column(db.Integer)
	param2 = db.Column(db.Integer)
	param3 = db.Column(db.Integer)
	param4 = db.Column(db.Integer)
	param5 = db.Column(db.Integer)
	param6 = db.Column(db.Integer)
	param7 = db.Column(db.Integer)
	param8 = db.Column(db.Integer)
	param9 = db.Column(db.Integer)
	param10 = db.Column(db.Integer)
	param11 = db.Column(db.Integer)
	param12 = db.Column(db.Integer)
	result = db.Column(db.Integer)

	def __init__(self,param1,param2,param3,param4,param5,param6,param7,param8,param9,param10,param11,param12,result):
		self.param1 = param1
		self.param2 = param2
		self.param3 = param3
		self.param4 = param4
		self.param5 = param5
		self.param6 = param6
		self.param7 = param7
		self.param8 = param8
		self.param9 = param9
		self.param10 = param10
		self.param11 = param11
		self.param12 = param12
		self.result = result


