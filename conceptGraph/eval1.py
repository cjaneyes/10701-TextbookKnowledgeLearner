import re 

class Eval1:
	def __init__(self,train_file,test_file):
		self.train_lines = open(train_file,"r").readlines()
		self.test_lines = open(test_file,"r").readlines()
		assert len(self.train_lines) == len(self.test_lines)
		self.recall = self.calculate_recall()
		#self.jaccard = self.jaccard()

	def calculate_recall(self):
		length = len(self.train_lines)
		recall_scores = []
		for i in range(length):
			test_concept = self.test_lines[i].split("\t")
			train_concept = self.train_lines[i].split("\t")
			score = self.recall(test_concept,train_concept)
			recall_scores.append(score)

		return sum(recall_scores) / float(len(recall_scores))


	def recall(self,test_concept,train_concept):
		train_dict = {}
		test_dict = {}
		for key in train_concept:
			re.sub("[0-9]","",key)
			if key in train_dict:
				train_dict[key] +=1 
			else:
				train_dict[key] = 1
		for key in test_concept:
			re.sub("[0-9]","",key)
			if key in test_dict:
				test_dict[key] += 1
			else:
				test_dict[key] = 1
		total_num = 0
		num_not_covered = 0
		for key in train_dict:
			num_train = train_dict[key]
			total_num += num_train
			num_not_covered += max(num_train - test_dict.get(key,0) , 0)

		return 1.0 - float(num_not_covered)/float(total_num)

	#def calculate_jaccard():


	#def jaccard(test_concept,train_concept):

if __name__ == '__main__':
	eval1 = Eval1("./test/annotation.out","./test/concepts_knowledge.txt")
	print eval1.recall

