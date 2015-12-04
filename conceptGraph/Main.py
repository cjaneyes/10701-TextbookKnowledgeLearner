from Sentence import Sentence
from Lexicon import Lexicon

class Main:
	def __init__(self,test_file,lexicon):
		self.sentences = []
		lines = open(test_file,"r").readlines()
		for line in lines:
			line = line.strip()
			self.sentences.append(Sentence(line,lexicon))

	def output(self):
		for sen in self.sentences:
			sen.output()

if __name__ == '__main__':
	lexicon = Lexicon()
	m = Main("./test/demo.txt",lexicon)
	m.output()
