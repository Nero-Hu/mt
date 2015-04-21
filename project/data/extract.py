#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import codecs
import xml.etree.ElementTree as ET

def utf8read(f):
    return codecs.open(f, 'r', 'utf-8')

def utf8write(f):
	return codecs.open(f, 'w', 'utf-8')

def main():
	jpCorpus = utf8write("wiki.jp")
	enCorpus = utf8write("wiki.en")
	# Read all xml file
	for f in glob.glob("data/*/*.xml"):
		try:
			print f
			tree = ET.parse(f)
		except Exception, e:
			continue
		else:
			root = tree.getroot()
			for par in root.findall("./par"):
				for sen in par.findall("./sen"):
					j = sen.find("./j")
					e = [en for en in sen.findall("./e") if en.attrib["type"] == "check"]
					if len(e) != 0 and e[0].text:
						jpCorpus.write(j.text + "\n")
						enCorpus.write(e[0].text + "\n")
			for par in root.findall("./*par"):
				for sen in par.findall("./sen"):
					j = sen.find("./j")
					e = [en for en in sen.findall("./e") if en.attrib["type"] == "check"]
					if len(e) != 0 and e[0].text:
						jpCorpus.write(j.text + "\n")
						enCorpus.write(e[0].text + "\n")

	jpCorpus.close()
	enCorpus.close()	

if __name__ == '__main__':
	main()