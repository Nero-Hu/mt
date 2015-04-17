###Inflection

Please refer to [MT class][1] for specific description.

Data used is [here][2].

####ngram
This script simply a n-gram language model. If we didn't see n-gram in our statistics, we simply look to (n-1)-gram, until we find unigram, or just return the origin word if we didn't find unigram. 

The default settings is a bigram model, use `-n` to try higher n-gram model.

[1]:http://mt-class.org/jhu/hw5.html
[2]:https://catalog.ldc.upenn.edu/LDC2006T01
