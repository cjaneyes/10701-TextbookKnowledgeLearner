from Feature import Feature

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

def getSample(sentence):
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
					params.append((function, [params[i],params[j]]))

	samples = []
	for predicate in concept_list[3]:
		if predicate.param_num == 1:
			for param in params:
				samples.append((predicate, param))
		else:
			for i in range(0,len(params)):
				for j in range(i+1,len(params)):
					samples.append((predicate,[params[i],params[j]]))
	for sample in samples:
		if len(sample[1]) == 1:
			sam = Feature([sample[0],sample[1][0][0]], sentence)
			features = sam.generateFeature()
		else:
			sam = Feature([sample[0],sample[1][0][0],sample[1][1][0]], sentence)
			features = sam.generateFeature()
		#print sample, features
		#print sentence,sample#,token_id,features