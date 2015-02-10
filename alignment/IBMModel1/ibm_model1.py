#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict

#Shameless copy 
optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)

sys.stderr.write("Start training...\n")
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]

f_count = 0
e_count = 0
f_list = []
e_list = []
#Get word counts
for f,e in bitext:
	for f_word in set(f):
		if f_word not in f_list:
			f_count += 1
			f_list.append(f_word)
	for e_word in set(e):
		if e_word not in e_list:
			e_count += 1
			e_list.append(e_word)

#Mapping location in dictionary
e_index = {}
f_index = {}
for (n,f) in enumerate(f_list):
	f_index[f] = n

for (n,e) in enumerate(e_list):
	e_index[e] = n

#Construct array as e_count x f_count
#Probability P(e|f)
pef = [[float(0) for x in range(f_count)] for x in range(e_count)]
#Initialize parameters with uniform distribution
for (n,e) in enumerate(pef):
	for f in range(len(e)):
		pef[n][f] = float(1)/float(f_count)

#TODO, check converge, considering check the change rate to be smaller than 0.01 for every word
for it in xrange(1,100):
	# Initialize 
	#Count for C(e|f)
	cef = [[float(0) for x in range(f_count)] for x in range(e_count)]
	totalf = [float(0)] * (f_count)
	totale = [float(0)] * (e_count)
	for f,e in bitext:
		#Compute normalization
		for e_word in set(e):
			totale[e_index[e_word]] = float(0)
			for f_word in set(f):
				totale[e_index[e_word]] += pef[e_index[e_word]][f_index[f_word]]
		#Collect counts
		for e_word in set(e):
			for f_word in set(f):
				cef[e_index[e_word]][f_index[f_word]] += float(pef[e_index[e_word]][f_index[f_word]]) / float(totale[e_index[e_word]])
				totalf[f_index[f_word]] += float(pef[e_index[e_word]][f_index[f_word]]) / float(totale[e_index[e_word]])
	#Estimate probabilities
	for f_word in f_list:
		for e_word in e_list:
			#Prevent for case 0
			if totalf[f_index[f_word]] == 0:
				pef[e_index[e_word]][f_index[f_word]] = float(0)
			else:	
				pef[e_index[e_word]][f_index[f_word]] = float(cef[e_index[e_word]][f_index[f_word]])/float(totalf[f_index[f_word]])

# Output word transfer
for (f, e) in bitext:
	for (i, f_word) in enumerate(f):
		max_j = 0
		max_word = ''
		max_prob = float(0)
		for (j, e_word) in enumerate(e):
			if pef[e_index[e_word]][f_index[f_word]] > max_prob:
				max_j = j
				max_word = e_word
				max_prob = pef[e_index[e_word]][f_index[f_word]]
		sys.stdout.write("%i-%i " % (i,max_j))
	sys.stdout.write("\n")