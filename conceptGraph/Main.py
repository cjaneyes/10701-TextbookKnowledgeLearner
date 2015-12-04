from Sentence import Sentence
from Lexicon import Lexicon

class Main:
	def __init__(self,test_file,lexicon):
		self.sentences = []
		lines = open(test_file,"r").readlines()
		for line in lines:
			line = line.strip()
			self.sentences.append(Sentence(line,lexicon))

	def output(self,file_name):
		for sen in self.sentences:
			sen.output(file_name)

if __name__ == '__main__':
	lexicon = Lexicon()
	m = Main("./test/knowledge.txt",lexicon)
	m.output('./test/concepts.txt')
