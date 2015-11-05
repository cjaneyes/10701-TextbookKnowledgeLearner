import os
import sys
import codecs
import re

#this code will read a each sentence line by line and output an input for ML model
#the format is label \t <index1>:<value1> .... (sparse representation)
#Run by command: python getFeatures.py INPUT_DIRECTORY OUTPUT_FIlEPATH

if __name__ == '__main__':
	vocab = {}
	folder_path = sys.argv[1]
	write_path= sys.argv[2]
	write_file = open(write_path,"w")
	for class_name in os.listdir(folder_path):
		if "class" not in class_name:
			print class_name
			continue
		for chapter in os.listdir(folder_path+"/"+class_name):	
			file_path = folder_path+'/'+class_name+"/"+chapter
			if ".DS_Store" in file_path:
				continue
			if "not_star" in file_path:
				continue
			f = codecs.open(file_path, encoding = 'utf-8')
			for line in f:
				line = line.replace("**","")
				features = {}
				sentence_type = 0
				label = '-1'    # refer to false label
				token = line.rstrip().lstrip().split("\t")
				sentence = token[0]

				#get label for each sentence
				if len(token) > 1:
					sentence = token[1]
					if token[0] == '-true':
						label = "1"    # refer to true label
					else:
						label = "-1"    # refer to notsure label

				#set the sentence type: 1 for questions 2 for declarative sentence 3 for exclaimation
				#sentence type is treated as the first feature
				features[1] = len(sentence)
				last_char = sentence[len(sentence)-1]
				if last_char == '.':
					features[2] = 1
				else:
					features[2] = 0
				if last_char == "?":
					features[3] = 1
				else:
					features[3] = 0
				if "," in sentence:
					features[4] = 1
				else:
					features[4] = 0
				if ":" in sentence:
					features[5] = 1
				else:
					features[5] = 0
				#only consider words that include [0-9a-zA-Z]
				sentence = re.sub("\\W+", " ", sentence, flags = re.UNICODE)
				words = sentence.split()
				if len(words) < 1:
					continue
				for w in words:
					if w not in vocab.keys():
						vocab[w] = len(vocab) + 5 + 1
					index = vocab[w]
					if index in features.keys():
						features[index] += 1
					else:
						features[index] = 1
				#write_file.write(label+"\t1:"+str(sentence_type)+" ")
				for key in sorted(features):
					write_file.write(str(key)+':'+str(features[key])+" ")
				write_file.write("\n")
