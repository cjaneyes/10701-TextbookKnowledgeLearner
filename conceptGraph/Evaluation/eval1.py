import re 

class Eval1:
	def __init__(self,train_file,test_file):
		self.train_lines = open(train_file,"r").readlines()
		self.test_lines = open(test_file,"r").readlines()
		assert len(self.train_lines) == len(self.test_lines)
		self.recall_score = self.calculate_recall()
		self.jaccard_score = self.calculate_jaccard()

	def calculate_recall(self):
		length = len(self.train_lines)
		recall_scores = []
		for i in range(length):
			test_concept = self.test_lines[i].strip().split("\t")
			train_concept = self.train_lines[i].strip().split("\t")
			score = self.recall(train_concept,test_concept)
			recall_scores.append(score)

		return sum(recall_scores) / float(len(recall_scores))


	def recall(self,train_concept,test_concept):
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

	def calculate_jaccard(self):
		length = len(self.train_lines)
		jaccard_scores = []
		for i in range(length):
			test_concept = self.test_lines[i].strip().split("\t")
			train_concept = self.train_lines[i].strip().split("\t")
			score = self.jaccard(train_concept,test_concept)
			jaccard_scores.append(score)

		return sum(jaccard_scores) / float(len(jaccard_scores))


	def jaccard(self,train_concept,test_concept):
		recall_rate = self.recall(train_concept,test_concept)
		size_train = len(train_concept) 
		size_test = len(test_concept)
		num_intersect = recall_rate * size_train
		num_union = size_train + size_test - num_intersect
		return float(num_intersect)/float(num_union)



if __name__ == '__main__':
	eval1 = Eval1("./demo.train","./demo.test")
	print eval1.recall_score
	print eval1.jaccard_score

