import os
import math

class Feature():
	def __init__(self, sample, sentence):
		self.parent = sample[0]
		self.children = sample[1:]
		self.sentence = sentence
		self.pos_set = []
		self.pos_map = self.genPOSmapping()
		self.feature = []

	def genPOSmapping(self):
		pos_map = {}
		count = 0
		for i in range(len(self.pos_set)):
			for j in range(len(self.pos_set)):
				pos_map[(self.pos_set[i], self.pos_set[j])] = count
				count += 1
		#print pos_map
		return pos_map


	def getDependencyTreeDist(self, pToken, cToken):
		queue = []
		queue.append([pToken, 0])
		while len(queue) > 0:
			ele = queue[0][0]
			depth = queue[0][1]
			if ele.id == cToken.id:
				break
			del queue[0]
			queue.append([self.sentence.tokens[ele.head], depth+1])
			for child in ele.children:
				queue.append([self.sentence.tokens[child], depth+1])
		return depth


	def getWordDist(self, pToken, cToken):
		value = math.fabs(pToken.id - cToken.id)
		return value


	def getDependencyTreeEdge(self, pToken, cToken):
		queue = []
		queue.append([pToken, 0])
		while len(queue) > 0:
			ele = queue[0][0]
			depth = queue[0][1]
			if ele.id == cToken.id:
				break
			del queue[0]
			for child in ele.children:
				queue.append([self.sentence.tokens[child], depth+1])
		if len(queue) == 0:
			depth = -1

		return depth


	# def getPOStag(self, pToken, cToken):
	# 	value = self.pos_map[(pToken.pos, cToken.pos)]
	# 	return value


	def getRelationType(self, pToken, cToken):
		value = 0.0
		return value


	def getReturnType(self, pToken, cToken):
		value = 0.0
		return value


	def generateFeature(self):
		#print self.parent
		pToken = self.sentence.tokens[self.parent.token_id]
		#print pToken
		for child in self.children:
			cToken = self.sentence.tokens[child.token_id]
			self.feature.append(self.getDependencyTreeDist(pToken, cToken))
			self.feature.append(self.getWordDist(pToken, cToken))
			self.feature.append(self.getDependencyTreeEdge(pToken, cToken))
			#self.feature.append(self.getRelationType(pToken, cToken))
			#self.feature.append(self.getReturnType(pToken, cToken))
		print self.feature
		return self.feature