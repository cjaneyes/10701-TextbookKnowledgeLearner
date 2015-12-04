from Gold_Concepts import Gold
from Concept import Concept
import copy
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
			if token.content.lower() in concept.keywords.split(","):
				tmp = Concept(concept,token)
				if tmp.type==1:
					for num in range(token.quant):
						concept_list.append(copy.copy(tmp))
				else:
					concept_list.append(tmp)
		return concept_list

