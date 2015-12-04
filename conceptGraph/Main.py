from Sentence import Sentence
class Main:
	def __init__(self,test_file):
		sentences = []
		lines = open(test_file,"r").readlines()
		for line in lines:
			line = line.strip()
			sentences.append(Sentence(line))

	def output(self):
		for sen in sentences:
			sen.output()

if __name__ == '__main__':
	m = Main("./test/test.raw")
	m.output()
