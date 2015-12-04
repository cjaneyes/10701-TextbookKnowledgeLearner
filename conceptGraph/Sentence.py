class Sentence:
	def __init__(self,text,lexicon):
		self.text = text
		tokens = []
		self.concepts = self.concepts_match(lexicon)
		self.update_variables(self.concepts)

	def get_dependency_tree(self):



		for token_id in range(1,len()):
			token = Token(token_id,___[token_id])
			self.tokens.append(token)

	def concepts_match(self,lexicon):
		for token in tokens:
			# update the concepts of token and return a list of concepts
			concepts = token.match_concepts(lexicon)
			self.concepts.append(concepts)

	def update_variables(self):
		name_dict = {}
		for concept in self.concepts:
			if concept.ctype == 1:
				name = concept.name
				if name not in name_dict:
					name_dict[name] = 1
				else:
					name_dict[name] += 1
				concept.name = concept.name + str(name_dict[name])


	def label_if_then(self):


	def relation_map(self):


	def output(self):
