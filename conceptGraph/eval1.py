class Eval1:
	def __init__(self,train_file,test_file):
		train_lines = open(train_file,"r").readlines()
		test_lines = open(test_file,"r").readlines()
		assert len(train_lines) == len(test_lines)


	def calculate_recall(self):
		length = len(train_lines)
		recall_scores = []
		for i in range(length):
			test_concept = test_line[i].split("\t")
			train_concept = train_line[i].split("\t")
			score = recall(test_concept,train_concept)
			recall_scores.append(score)

		return sum(recall_scores) / float(len(recall_scores))


	def recall(test_concept,train_concept):
		train_dict = {}
		test_dict = {}
		for key in train_concept:
			if key in train_dict:
				train_dict[key] +=1 
			else:
				train_dict[key] = 1
		for key in test_concept:
			if key in test_dict:
				test_dict[key] += 1
			else:
				test_dit[key] = 1
		total_num = 0
		num_not_covered = 0
		for key in train_key:
			num_train = train_dict[key]
			total_num += num_train
			num_not_covered += max(num_train - test_dict.get(key,0) , 0)

		return 1.0 - float(num_not_covered)/float(total_num)

	def calculate_jaccard():


	def jaccard(test_concept,train_concept):


