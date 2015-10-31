import nltk.data
import os

class sentenceTokenizer():
	def __init__(self,path):
		# parsing folder is not included now
		self.file_name = path.split("/")[-1]
		text = open(path,"r").read()
		self.text = text.decode("utf8").replace("\n"," ").strip()
		sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
		self.sentences = sent_tokenizer.tokenize(self.text)

	def write2file(self,folder):
		write_file = open("./"+folder+"/"+self.file_name,"w")
		for sen in senTokenizer.sentences:
			write_file.write(sen.encode("utf8")+"\n")


if __name__=='__main__':
	# get all txt files from data folder
	folder_path = "../data"
	for folder in os.listdir(folder_path):
		if "class" in folder:
			book_path = folder_path + "/" + folder
			for chapter in os.listdir(book_path):
				chapter_path = book_path + "/" + chapter
				if ".txt" in chapter_path:
					if not os.path.exists(folder):
						os.makedirs(folder)
					senTokenizer = sentenceTokenizer(chapter_path)
					senTokenizer.write2file(folder)
