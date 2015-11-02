import os
import re
# heuristic ways of preprocessing sentence level text
class processer:
	def __init__(self,path,output_path):
		# return a dict that describes all text
		# first key is the name of the chapter
		folder_path = path
		self.inputpath = path
		self.outputpath = output_path
		self.text = {}
		for folder in os.listdir(folder_path):
			if "class" in folder:
				book_path = folder_path + "/" + folder
				for chapter in os.listdir(book_path):
					chapter_path = book_path + "/" + chapter
					if ".txt" in chapter_path:
						#print chapter_path
						self.text[chapter_path] = []
						lines = open(chapter_path,"r").read().replace("\r","\n").split("\n")
						# handle annotation 
						for i in range(len(lines)):
							line = lines[i].strip().split("\t")
							if line[0] == "-true":
								label ="-true\t"
							else:
								label = ""
							lines[i] = lines[i].replace("-true\t","")

							# delete sentences that are too short 
							if  self.handleShort(lines[i]):
								#self.handleFilename(lines[i])
								rmp_line = self.handlePrefix(lines[i])
								self.text[chapter_path].append("**"+label+ rmp_line+"**\n")
		#self.write2file(output_path)


	# note that we need to convert utf8 to unicode for processing and convert back to utf8 before output
	
	def handleFilename(self,line):
		re_file = []
		re_file.append(re.compile(r'File(.)*PM65'))
		re_file.append(re.compile(r'File(.)*pmd')) 
		for i in range(len(re_file)):
			tmp = re_file[i].findall(line)
			if tmp:
				print tmp
				line = line.replace(tmp,"")
				print line
		return line



	def handlePrefix(self,line):
		# find words all capital , delete characters until we find another Captial.
		re_capital = re.compile(r'[A-Z]{5,20}')
		words = line.split(" ")
		for word in words:
			word = word.strip()
			if re.match(re_capital,word):
				line = line.replace(word,"")
		# sentence must start with capital! ..
		m = re.search("[A-Z]", line)
		if m:
			line = line[m.start():]
		return line.strip()


	def handleShort(self,line):
		# handle too short sentences. (short means len<=20)
		if len(line)<=20:
			return 0
		else:
			return 1

	#def handleNoise(self,line):
		# sentence level
	#	return 1
		# handle the problem of meaningless symbols

	def write2file(self,path):
		for key in self.text.keys():
			[dot, folder,book,chapter] = key.split("/")
			folder = path.replace(self.inputpath,self.outputpath)

			if not os.path.exists(folder):
				os.makedirs(folder)
			if not os.path.exists(folder+"/"+book):
				os.makedirs(folder+"/"+book)
			file_path = folder+"/"+book+"/"+chapter
			write_file = open(file_path,"w")
			for line in self.text[key]:
				write_file.write(line)

if __name__ == '__main__':
	p = processer("./sentences","./clean_sentences")
	#test_line = "TRIANGLES  123  6.3 Similarity of Triangles What can you say about the similarity of two triangles?\n"
	#print p.handlePrefix(test_line)
	p.write2file(p.outputpath)

