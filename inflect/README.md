###Inflection

Please refer to [MT class][1] for specific description.

Data used is [here][2].

####ngram
This script is simply a n-gram language model. If we didn't see n-gram in our statistics, we simply look to (n-1)-gram, until we find unigram, or just return the origin word if we didn't find unigram. 

The default settings is a bigram model, use `-n` to try higher n-gram model.

####ngram_pos
This script is built based on `ngram` model, but encorporate Part-of-Speech information. Now we use `joint` n-gram based on `lemma` and `Part-of-Speech`, which means we will look at the ngram, then it's pos ngram, if nither of them exists then we just backoff to lower gram. Only special is that if we didn't see any unigram co-occured with a `POS` tag, we simply use its unigram.

The default settings is a trigram model, use `-n` to try higher n-gram model.

####bigram_dep
This parse the dependency information in a bigram model. The back-off scheme reduces to tagged unigram, and unigram at last.

[1]:http://mt-class.org/jhu/hw5.html
[2]:https://catalog.ldc.upenn.edu/LDC2006T01
