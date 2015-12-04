class Concept:
	def __init__(self, gold_concept,token):
		self.name = gold_concept.name
		self.type = gold_concept.type # 0 for constant , 1 for variable, 2 for function and 3 for predicate
		self.token_id = token.id
		self.param_num = gold_concept.param_num
		self.if_then = token.if_then

