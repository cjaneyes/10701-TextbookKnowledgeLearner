class Token:
	def __init__(self, token_id, diction):
        self.id = token_id
        self.head =  diction["head"]
        self.content = diction["content"]
        self.pos = diction["pos"]
        self.if_then = dictiom["if_then"]
        self.concepts = []
        self.children = diction["children"]


	def match_concepts(self,lexicon):
		# return concepts list based on token (word, and depend tree info)
		concepts_list = lexicon.match_concepts(token)
		self.concepts = concepts_list
		return concepts_list
