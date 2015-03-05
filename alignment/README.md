###Word Alignment
An implementation of word aligner, based on [fast align][1].
The check and score scripts from [Philipp Koehn][2].

Following command will run the program with 1000 lines of data, and print out the score of golden 37 sentences.

    python fast_aligh.py -n 1000 | ./score-alignments

[1]:http://aclweb.org/anthology/N/N13/N13-1073.pdf
[2]:http://mt-class.org/jhu/hw1.html
