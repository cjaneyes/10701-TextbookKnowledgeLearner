import nltk
from nltk.parse.stanford import StanfordDependencyParser
from nltk.parse import DependencyGraph
from Lexicon import Lexicon
from Token import Token
from if_then_parsing import if_then_parsing


class Sentence:
	def __init__(self,text,lexicon):
		self.text = text
		self.get_dependency_tree()
		self.concepts = []
		self.update_quant()
		self.concepts_match(lexicon)
		self.update_variables()
		print "process"

	def get_dependency_tree(self):

		sentence = if_then_parsing(self.text)
		self.logic_text = sentence
		#path_to_jar = '/Users/jane_C/Documents/CMU/Courses/10701-MachineLearning/project/KnowledgeLearning/lib/stanford-parser/stanford-parser.jar'
		#path_to_models_jar = '/Users/jane_C/Documents/CMU/Courses/10701-MachineLearning/project/KnowledgeLearning/lib/stanford-parser/stanford-parser-3.5.2-models.jar'

		path_to_jar = '../lib/stanford-parser/stanford-parser.jar'
		path_to_models_jar = '../lib/stanford-parser/stanford-parser-3.5.2-models.jar'
		
		dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)

		sentence_parse = dependency_parser.raw_parse(sentence)

		tokenList = []
		tokenInfo = {}
		tokenInfo["content"] = "ROOT"
		tokenInfo["pos"] = "ROOT"
		tokenInfo["head"] = -1
		tokenInfo["children"] = []
		tokenInfo["if_then"] = -1
		root = Token(0, tokenInfo)
		tokenList.append(root)

		left2right = True
		left2right_point = -1
		index = 0
		for sent in sentence_parse:
			sent_conll = sent.to_conll(10)
			tokens = sent_conll.split("\n")
			index = 0
			for term in tokens:
				index += 1
				tokenInfo = {}
				parse = term.strip().split("\t")
				if term == "" or len(parse) < 10:
					continue
				if parse[1] == ">" or parse[1] == "<":
					if parse[1] == "<":
						left2right = False
					left2right_point = index
					#continue
				tokenInfo["content"] = parse[1]
				tokenInfo["pos"] = parse[4]
				tokenInfo["head"] = int(parse[6])
				tokenInfo["children"] = []
				tokenInfo["if_then"] = 0
				t = Token(index, tokenInfo)
				tokenList.append(t)

		if left2right:
			for i in range(left2right_point, len(tokenList)):
				tokenList[i].if_then = 1
		else:
			for i in range(1, left2right_point):
				tokenList[i].if_then = 1
		tokenList[left2right_point].if_then = -1
		for i in range(1, len(tokenList)):
			token = tokenList[i]
			tokenList[token.head].children.append(i)

		self.tokens = tokenList


	def concepts_match(self,lexicon):
		for token in self.tokens:
			# update the concepts of token and return a list of concepts
			concepts = token.match_concepts(lexicon)
			if concepts!=[]:
				self.concepts.append(concepts)		

	def update_variables(self):
		name_dict = {}
		for concepts in self.concepts:
			for concept in concepts:
				if concept.type == 1:
					name = concept.name
					if name not in name_dict:
						name_dict[name] = 1
					else:
						name_dict[name] += 1
					concept.name = concept.name + str(name_dict[name])

	def update_quant(self):
		for token in self.tokens:
			for child in token.children:
				child_content = self.tokens[child].content.lower()
				if "two" in child_content or "both" in child_content:
					token.quant = 2
				elif "three" in child_content:
					token.quant = 3
				elif "all" in child_content and "triangle" in token.content:
					token.quant = 3


	def output(self,file_name):
		write_file = open(file_name,"a")
		print self.logic_text
		for token in self.tokens:
			for concept in token.concepts:
				write_file.write(concept.name + "\t")
				print token.content + " is " + concept.name + "\t" + str(concept.type) ,
				if concept.type == "3":
					print 'predicate	',
				elif concept.type =="2":
					print "function	",
				else:
					print "variable	",
				print  " => ",
				if token.if_then==1:
					print "Then"
				else:
					print "If" 

		write_file.write("\n")

		print "=== end of this sentence ===="



