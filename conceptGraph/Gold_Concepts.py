class Gold:
	def __init__(self, name, ctype, param_num, keywords):
		self.name = name
		self.type = ctype # 0 for constant , 1 for variable, 2 for function and 3 for predicate
		self.param_num = param_num
		self.keywords = keywords