###A standard porter stemmer

diff.txt contains the original and after-stemmer pair.

Run following to check the stemmer accuracy.

	./stemmer.py
	
####How to use

Put ``stemmer.py`` into your working directory.

	import stemmer
	
    """
    Test with porter stemming data pairs
    """
    corr = 0.0
    count = 0.0
    stm = stemmer.Stemmer()
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