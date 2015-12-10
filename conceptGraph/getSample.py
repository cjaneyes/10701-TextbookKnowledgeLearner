from Feature import Feature

def convert_features(features):
	length = len(features)
	return_str = ""
	for i in range(length):
		return_str += str(i)+":"+str(features[i])+" "
	return return_str

def reorder_concepts(concepts):
	concept_list = []
	for i in range(0,4):
		tmp = []
		concept_list.append(tmp)

	for token_concept in concepts:
		for concept in token_concept:
			concept_list[concept.type].append(concept)
	return concept_list

def remove_duplicate(concept_list):
	predicates = {}
	for concept in concept_list[3]:
		predicates[concept.name] = concept
	concept_list[3] = []
	for k,v in predicates:
		concept_list[3].append(v)
	return concept_list

def function_filter(function, param1, param2):
	param1 = param1[0]
	param2 = param2[0]
	if param1.type == 1 and param2.type == 1:
		if param1.name[:len(param1.name)-1] != param2.name[:len(param2.name)-1]:
			return True
	return False

def binary_filter(predicate, param1, param2):
	param1 = param1[0]
	param2 = param2[0]
	var_var_equals_filter = ['Equals', 'LessThan', 'GreaterThan','Congruent','Similar','IsHypotenuse','IsLeg']
	if(predicate.name == 'AngleOf'):
		if param1.name[:5] != 'angle':
			return True
		elif param2.name[:5] == 'angle':
			return True
	if predicate.name in var_var_equals_filter:
		if param1.type == 0 and param1.name != 'Equals':
			return True
		if param1.type == 1 and param2.type == 1:
			if param1.name[:len(param1.name)-1] != param2.name[:len(param2.name)-1]:
				return True
	return False

def order_matter(predicate):
	matter = ['LessThan', 'SmallerThan', 'GreaterThan']
	if predicate.name in matter:
		return True
	if predicate.name[len(predicate.name)-2:] == 'Of':
		return True
	return False

def getSample(sentence):
	write_file_1 = open("uni.test.literal","a")
	write_file_2 = open("bi.test.literal","a")
	concept_list = reorder_concepts(sentence.concepts)
	params = []
	for concept in concept_list[0]:
		params.append((concept,[]))
	for concept in concept_list[1]:
		params.append((concept,[]))
	for function in concept_list[2]:
		if function.param_num == 1:
			for param in params:
				params.append((function, [param]))
		else:
			for i in range(0,len(params)):
				for j in range(i+1,len(params)):
					if not function_filter(function, params[i], params[j]):
						params.append((function, [params[i],params[j]]))

	samples = []
	for predicate in concept_list[3]:
		if predicate.param_num == 1:
			for param in params:
				samples.append((predicate, [param])) # lack [] initially
		else:
			for i in range(0,len(params)):
				for j in range(i+1,len(params)):
					if not binary_filter(predicate, params[i], params[j]):
						samples.append((predicate,[params[i],params[j]]))
					if order_matter(predicate):
						if not binary_filter(predicate, params[j], params[i]):
							samples.append((predicate,[params[j],params[i]]))

	for sample in samples:
		print sentence.text
		print sample[0].name+"(",
		if len(sample[1]) == 1:
			print sample[1][0][0].name+  "-" +str(sample[1][0][0].token_id) +")"
			sam = Feature([sample[0],sample[1][0][0]], sentence)
			features = sam.generateFeature()
			write_file_1.write(sentence.text+"\n"+sample[0].name + "-" + str(sample[0].token_id) + "(" + sample[1][0][0].name+  "-" +str(sample[1][0][0].token_id) +")\t"+convert_features(features)+"\n")

		else:
			print sample[1][0][0].name + "-" + str(sample[1][0][0].token_id) +","+sample[1][1][0].name + "-" + str(sample[1][1][0].token_id)+")"
			sam = Feature([sample[0],sample[1][0][0],sample[1][1][0]], sentence)
			features = sam.generateFeature()
			write_file_2.write(sentence.text+"\n"+sample[0].name+ "-" + str(sample[0].token_id)+"("+sample[1][0][0].name + "-" + str(sample[1][0][0].token_id) +","+sample[1][1][0].name + "-" + str(sample[1][1][0].token_id)+")\t"+convert_features(features)+"\n")

		#print sample, features
		print features
		#print features