import re

def delete_colon(line):
	tokens = line.split(':')
	return tokens[1]

def with_then_marking(line, mark):
	tokens = line.split(mark)
	return tokens[0]+' > '+ mark+tokens[1]

def with_if_marking(line, mark):
	tokens = line.split(mark)
	return tokens[0]+' < '+mark+tokens[1]

def contains_if(line):
	tokens = line.split('if ')
	if len(tokens) == 2 and ',' not in tokens[1]:
		return tokens[0]+' < '+'if '+ tokens[1]
	else:
		tokens = line.split('are ')
		return tokens[0]+' > are '+tokens[1]

def contains_are(line):
	tokens = line.split(' are ')
	res = ""
	for i in range(0,len(tokens)):
		if i == len(tokens)-1:
			res += ' > '+tokens[i]
		else:
			res += tokens[i]+' are '
	return res

def contains_is(line):
	tokens = line.split('is ')
	res = ""
	for i in range(0, len(tokens)):
		if i == 0:
			res += tokens[i]+' > '
		else:
			res += 'is '+tokens[i]
	return res

def contains_comma(line):
	tokens = line.split(',')
	return tokens[0]+' > ,'+tokens[1]

def contains_measures(line):
	tokens = line.split('measures')
	return tokens[0]+' > measures'+tokens[1]

def contains_bisect(line):
	tokens = line.split('bisect')
	return tokens[0]+' > bisect'+tokens[1]

def if_then_parsing(line):
	then_markers = ['then ', 'is called ', 'hence ', 'are called']
	if_markers = [', if']
	line = re.sub('[^0-9a-zA-Z\.,:?!;=\-\s]+','', line)
	line = line.strip().lower()
	if ':' in line:
		line = delete_colon(line)
	for w in then_markers:
		if w in line:
			line = with_then_marking(line, w)
			line = re.sub(',','', line)
			return line
	for w in if_markers:
		if w in line:
			line = with_if_marking(line, w)
			line = re.sub(',','', line)
			return line
	if 'if ' in line:
		line = contains_if(line)
		line = re.sub(',','', line)
		return line
	if ' are ' in line:
		line = contains_are(line)
		line = re.sub(',','', line)
		return line
	if 'is ' in line:
		line = contains_is(line)
		line = re.sub(',','', line)
		return line
	if ',' in line:
		line = contains_comma(line)
		line = re.sub(',','', line)
		return line
	if 'measures' in line:
		line = contains_measures(line)
		line = re.sub(',','', line)
		return line
	if 'bisect' in line:
		line = contains_bisect(line)
		line = re.sub(',','', line)
		return line