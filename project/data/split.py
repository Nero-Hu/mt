#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs

def utf8read(f):
    return codecs.open(f, 'r', 'utf-8')

def utf8write(f):
	return codecs.open(f, 'w', 'utf-8')

def main():
	jpTrain = utf8write("wiki-train.jp")
	enTrain = utf8write("wiki-train.en")
	jpTest = utf8write("wiki-test.jp")
	enTest = utf8write("wiki-test.en")
	# Read all xml file
	i = 0
	for en, jp in zip(utf8read("wiki.en"), utf8read("wiki.jp")):
		if i < 350000:
			jpTrain.write(jp)
			enTrain.write(en)
		else:
			jpTest.write(jp)
			enTest.write(en)
		i += 1

	jpTrain.close()
	jpTest.close()
	enTrain.close()
	enTest.close()	

if __name__ == '__main__':
	main()