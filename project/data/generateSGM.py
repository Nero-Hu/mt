#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import codecs
import xml.etree.ElementTree as ET

from xml.dom import minidom

def utf8read(f):
    return codecs.open(f, 'r', 'utf-8')

def utf8write(f):
	return codecs.open(f, 'w', 'utf-8')

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="")

def main():
	jpSgm = utf8write("wiki-test.jp.sgm")
	enSgm = utf8write("wiki-test.en.sgm")
	srcRoot = ET.Element("srcset")
	refRoot = ET.Element("refset")
	srcRoot.set("srclang", "any")

	srcDoc = ET.SubElement(srcRoot, "doc")
	refDoc = ET.SubElement(refRoot, "doc")
	i = 1
	for jp, en in zip(utf8read("wiki-test.jp"), utf8read("wiki-test.en")):
		jp = jp.rstrip()
		en = en.rstrip()
		ET.SubElement(srcDoc, "seg", id=str(i)).text =jp
		ET.SubElement(refDoc, "seg", id=str(i)).text =en

		i += 1

	jpSgm.write(prettify(srcRoot))
	enSgm.write(prettify(refRoot))

	jpSgm.close()
	enSgm.close()	

if __name__ == '__main__':
	main()