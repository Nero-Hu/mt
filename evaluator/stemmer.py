#!/usr/bin/env python

class Stemmer:
	"""
	An implementation for port stemmer

	reference: 
	http://snowball.tartarus.org/algorithms/porter/stemmer.html

	"""
	def __init__(self):
		self.vowel = ['a','e','i','o','u','y']

	def stem(self, word):
		"""
		word, string: the word to be stemmed
		"""
		# Consective VC pairs
		self.tagList = [None] * len(word)
		res = word

		word = word.strip().lower()
		# 1 length word is trivial
		if len(word) == 1:
			return word

		#Analyze word structure
		for i in range(len(word)):
			if word[i] not in self.vowel:
				self.tagList[i] = 'C'
			#Note that if y is the first letter, it considered as consonent
			elif word[i] == 'y' and ((i >= 1 and word[i-1] in self.vowel) or i == 0):
				self.tagList[i] = 'C'
			else:
				self.tagList[i] = 'V'

		# Start to process
		res = self.step1a(res)
		res = self.step1b(res)
		res = self.step1c(res)
		res = self.step2(res)
		res = self.step3(res)
		res = self.step4(res)
		res = self.step5a(res)
		res = self.step5b(res)
		return res

	def step1a(self, word):
		l = len(word)
		if l >=4 and word[l-4:] == 'sses':
			word = word[:l-2]
		elif l >= 3 and word[l-3:] == 'ies':
			word = word[:l-2]
		elif l >= 2 and word[l-2:] == 'ss':
			word = word
		elif l >=2 and word[l-1:] == 's':
			word = word[:l-1]

		return word

	def step1b(self, word):
		match1b = False
		l = len(word)
		if word[l-3:] == 'eed':
			if self.computeM(l-3) > 0 and l >= 3:
				word = word[:l-1]
		elif l >= 2 and word[l-2:] == 'ed':
			if 'V' in self.tagList[:l-2]:
				match1b = True
				word = word[:l-2]
		elif l >= 3 and word[l-3:] == 'ing':
			if 'V' in self.tagList[:l-3]:
				match1b = True
				word = word[:l-3]

		#If get here, then 2/3 has matched
		if match1b:
			l = len(word)
			if l >= 2 and word[l-2:] == 'at':
				word += 'e'
			elif l >= 2 and word[l-2:] == 'bl':
				word += 'e'
			elif l >= 2 and word[l-2:] == 'iz':
				word += 'e'
			elif l >= 2 and word[l-2] == word[l-1] and word[l-1] not in ['l','s','z'] and word[l-1] not in self.vowel:
				word = word[:l-1]
			elif l >= 3 and self.computeM(l) == 1 and self.isCVC(word, l-1):
				word += 'e'

		return word

	def step1c(self, word):
		l = len(word)
		if word[l-1] == 'y' and 'V' in self.tagList[:l-1]:
			word = word[:l-1] +'i'
		return word

	def step2(self, word):
		l = len(word)
		if l >= 7 and self.computeM(l-7) >0 and word[l-7:] == 'ational':
			word = word[:l-5] + 'e'
		elif l >= 6 and self.computeM(l-6) >0 and word[l-6:] == 'tional':
			word = word[:l-2]
		elif l >= 4 and self.computeM(l-4) >0 and word[l-4:] == 'enci':
			word = word[:l-1] + 'e'
		elif l >= 4 and self.computeM(l-4) >0 and word[l-4:] == 'anci':
			word = word[:l-1] + 'e'
		elif l >= 4 and self.computeM(l-4) >0 and word[l-4:] == 'izer':
			word = word[:l-1]
		elif l >= 4 and self.computeM(l-4) >0 and word[l-4:] == 'abli':
			word = word[:l-1] + 'e'
		elif l >= 4 and self.computeM(l-4) >0 and word[l-4:] == 'alli':
			word = word[:l-2]
		elif l >= 5 and self.computeM(l-5) >0 and word[l-5:] == 'entli':
			word = word[:l-2]
		elif l >= 3 and self.computeM(l-3) >0 and word[l-3:] == 'eli':
			word = word[:l-2]
		elif l >= 5 and self.computeM(l-5) >0 and word[l-5:] == 'ousli':
			word = word[:l-2]
		elif l >= 7 and self.computeM(l-7) >0 and word[l-7:] == 'ization':
			word = word[:l-5] + 'e'
		elif l >= 5 and self.computeM(l-5) >0 and word[l-5:] == 'ation':
			word = word[:l-3] + 'e'
		elif l >= 4 and self.computeM(l-4) >0 and word[l-4:] == 'ator':
			word = word[:l-2] + 'e'
		elif l >= 5 and self.computeM(l-5) >0 and word[l-5:] == 'alism':
			word = word[:l-3]
		elif l >= 7 and self.computeM(l-7) >0 and word[l-7:] == 'iveness':
			word = word[:l-4]
		elif l >= 7 and self.computeM(l-7) >0 and word[l-7:] == 'fulness':
			word = word[:l-4]
		elif l >= 7 and self.computeM(l-7) >0 and word[l-7:] == 'ousness':
			word = word[:l-4]
		elif l >= 5 and self.computeM(l-5) >0 and word[l-5:] == 'aliti':
			word = word[:l-3]
		elif l >= 5 and self.computeM(l-5) >0 and word[l-5:] == 'iviti':
			word = word[:l-3] + 'e'
		elif l >= 6 and self.computeM(l-6) >0 and word[l-6:] == 'biliti':
			word = word[:l-5] + 'le'

		return word

	def step3(self, word):
		l = len(word)
		if l >= 5 and self.computeM(l-5) >0 and word[l-5:] == 'icate':
			word = word[:l-3]
		elif l >= 5 and self.computeM(l-5) >0 and word[l-5:] == 'ative':
			word = word[:l-5]
		elif l >= 5 and self.computeM(l-5) >0 and word[l-5:] == 'alize':
			word = word[:l-3]
		elif l >= 5 and self.computeM(l-5) >0 and word[l-5:] == 'iciti':
			word = word[:l-3]
		elif l >= 4 and self.computeM(l-4) >0 and word[l-4:] == 'ical':
			word = word[:l-2]
		elif l >= 3 and self.computeM(l-3) >0 and word[l-3:] == 'ful':
			word = word[:l-3]
		elif l >= 4 and self.computeM(l-4) >0 and word[l-4:] == 'ness':
			word = word[:l-4]

		return word

	def step4(self, word):
		l = len(word)
		if l >= 2 and self.computeM(l-2) >1 and word[l-2:] == 'al':
			word = word[:l-2]
		elif l >= 4 and self.computeM(l-4) >1 and word[l-4:] == 'ance':
			word = word[:l-4]
		elif l >= 4 and self.computeM(l-4) >1 and word[l-4:] == 'ence':
			word = word[:l-4]
		elif l >= 2 and self.computeM(l-2) >1 and word[l-2:] == 'er':
			word = word[:l-2]
		elif l >= 2 and self.computeM(l-2) >1 and word[l-2:] == 'ic':
			word = word[:l-2]
		elif l >= 4 and self.computeM(l-4) >1 and word[l-4:] == 'able':
			word = word[:l-4]
		elif l >= 4 and self.computeM(l-4) >1 and word[l-4:] == 'ible':
			word = word[:l-4]
		elif l >= 3 and self.computeM(l-3) >1 and word[l-3:] == 'ant':
			word = word[:l-3]
		#Caution, always match most, and only once
		elif word[l-5:] == 'ement':
			if l >= 5 and self.computeM(l-5) >1:
				word = word[:l-5]
		elif word[l-4:] == 'ment':
			if l >= 4 and self.computeM(l-4) >1:
				word = word[:l-4]
		elif l >= 3 and self.computeM(l-3) >1 and word[l-3:] == 'ent':
			word = word[:l-3]
		elif l >= 4 and self.computeM(l-3) >1 and word[l-4] in ['s','t'] and word[l-3:] == 'ion':
			word = word[:l-3]
		elif l >= 2 and self.computeM(l-2) >1 and word[l-2:] == 'ou':
			word = word[:l-2]
		elif l >= 3 and self.computeM(l-3) >1 and word[l-3:] == 'ism':
			word = word[:l-3]
		elif l >= 3 and self.computeM(l-3) >1 and word[l-3:] == 'ate':
			word = word[:l-3]
		elif l >= 3 and self.computeM(l-3) >1 and word[l-3:] == 'iti':
			word = word[:l-3]
		elif l >= 3 and self.computeM(l-3) >1 and word[l-3:] == 'ous':
			word = word[:l-3]
		elif l >= 3 and self.computeM(l-3) >1 and word[l-3:] == 'ive':
			word = word[:l-3]
		elif l >= 3 and self.computeM(l-3) >1 and word[l-3:] == 'ize':
			word = word[:l-3]

		return word

	def step5a(self, word):
		l = len(word)
		if l >= 1 and self.computeM(l-1) >1 and word[l-1:] == 'e':
			word = word[:l-1]
		elif l >= 1 and self.computeM(l-1) == 1 and not self.isCVC(word, l-2) and word[l-1:] == 'e':
			word = word[:l-1]

		return word

	def step5b(self, word):
		l = len(word)
		if self.computeM(l) > 1 and word[l-1] == word[l-2] and word[l-1] == 'l':
			word = word[:l-1]

		return word

	def isCVC(self, word, loc):
		"""
		Test if the word ends up at 'loc' is a CVC structure
		"""
		if loc < 2:
			return False
		
		return self.tagList[loc-2] == 'C' and self.tagList[loc-1] == 'V' and self.tagList[loc] == 'C' and word[loc] not in ['w','x','y']

	def computeM(self, loc):
		"""
		Compute VC pairs number giving the location
		"""
		m = 0
		tgList = self.tagList[:loc]
		for i in range(loc-1):
			if tgList[i] == 'V' and tgList[i+1] == 'C':
				m += 1

		return m

def main():
	"""
	Test with porter stemming data pairs
	"""
	corr = 0.0
	count = 0.0
	stm = Stemmer()
	with open("diffs.txt") as f:
		for line in f:
			count += 1
			line = line.strip().lower()
			ori, ans = line.split()
			res = stm.stem(ori)
			if res == ans:
				corr += 1
			else:
				print ori, ans, res

	print corr/count
    
if __name__ == '__main__':
    main()
