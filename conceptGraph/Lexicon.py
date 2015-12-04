from Gold_Concepts import Gold
class Lexicon:
	def __init__(self):
		# 0 for constant, 1 for variable , 2 for function and 3 for predicate
		lines = open("./training/clean_lexicon.txt","r").readlines()
		self.lexicon = []
		for line in lines:
			[cname, ctype, num_param, keywords] = line.strip().split("\t")
			c = Gold(cname,ctype,num_param,keywords)
			self.lexicon.append(c)

	def match_lexicon(self,token):
		concept_list = []
		for concept in self.lexicon:
			if token.content.lower() in concept.keywords:
				concept_list.append(Concept(concept,token))
		return concept_list

