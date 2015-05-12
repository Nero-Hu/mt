### RNN Phrase Encoder/Decoder

This repository	contains a RNN based phrase encoder/decoder.

The original paper is here [Learning phrase representations using rnn encoder-decoder for statistical machine translation][1].

### Usage

You need following to run this scripts.

- numpy
- scipy
- theano

Note that this scripts takes a long time to train, even on GPUs. Test has been done on a **GTX 690** GPU, and it takes around **14** hours to train, and **12** hours to rescore.

The output looks like following:

	Epochs 1, totalNLL: 489089.337824, used time: 870.954233
	Epochs 2, totalNLL: 388776.033000, used time: 867.024827
	Epochs 3, totalNLL: 326049.774204, used time: 863.530336
	Epochs 4, totalNLL: 313805.663657, used time: 863.959077
	Epochs 5, totalNLL: 304659.267954, used time: 863.861142
	Epochs 6, totalNLL: 299589.291673, used time: 863.427297
	Epochs 7, totalNLL: 294872.777107, used time: 863.887010
	Epochs 8, totalNLL: 291013.141479, used time: 863.973088
	Epochs 9, totalNLL: 287900.712534, used time: 863.731001
	Epochs 10, totalNLL: 283769.578505, used time: 864.008877
	Epochs 11, totalNLL: 279410.579484, used time: 863.814905
	Epochs 12, totalNLL: 275802.991499, used time: 863.789746
	Epochs 13, totalNLL: 273676.443551, used time: 863.834051
	Epochs 14, totalNLL: 272727.013496, used time: 863.384957
	Epochs 15, totalNLL: 270667.354443, used time: 863.982505
	Epochs 16, totalNLL: 270237.144082, used time: 863.946339
	Epochs 17, totalNLL: 269119.814058, used time: 863.588089
	Epochs 18, totalNLL: 269731.661034, used time: 864.040740
	Epochs 19, totalNLL: 269473.756211, used time: 863.982663
	Epochs 20, totalNLL: 265176.443387, used time: 863.713470
	Epochs 21, totalNLL: 262828.888353, used time: 863.758193
	Epochs 22, totalNLL: 262740.830963, used time: 863.703632
	Epochs 23, totalNLL: 263320.757356, used time: 863.380525
	Epochs 24, totalNLL: 261952.926632, used time: 863.840602
	Epochs 25, totalNLL: 261099.912897, used time: 863.196999
	Epochs 26, totalNLL: 258559.529776, used time: 863.493585
	Epochs 27, totalNLL: 257647.187679, used time: 863.629476
	Epochs 28, totalNLL: 256022.517993, used time: 863.246485
	Epochs 29, totalNLL: 256290.225152, used time: 863.345007
	Epochs 30, totalNLL: 254283.060196, used time: 863.488554
	Epochs 31, totalNLL: 255983.462444, used time: 863.321535
	Epochs 32, totalNLL: 257816.435385, used time: 863.304585
	Epochs 33, totalNLL: 254267.007380, used time: 863.559946
	Epochs 34, totalNLL: 253364.630339, used time: 863.002616
	Epochs 35, totalNLL: 253437.236206, used time: 863.207589
	Epochs 36, totalNLL: 255126.618918, used time: 863.477396
	Epochs 37, totalNLL: 261442.506691, used time: 863.288585
	Epochs 38, totalNLL: 257431.341616, used time: 863.181421
	Epochs 39, totalNLL: 255875.292801, used time: 863.402434
	Epochs 40, totalNLL: 253958.448351, used time: 862.841031
	Epochs 41, totalNLL: 251748.030944, used time: 863.159567
	Epochs 42, totalNLL: 250123.631277, used time: 863.400333
	Epochs 43, totalNLL: 249780.805722, used time: 862.932598
	Epochs 44, totalNLL: 249000.669947, used time: 863.364040
	Epochs 45, totalNLL: 249557.030181, used time: 863.383479
	Epochs 46, totalNLL: 247054.131129, used time: 863.134350
	Epochs 47, totalNLL: 246026.088518, used time: 863.423700
	Epochs 48, totalNLL: 244333.042093, used time: 863.296683
	Epochs 49, totalNLL: 245519.184605, used time: 863.209351
	Epochs 50, totalNLL: 245525.693625, used time: 863.488372

Time is calculated in seconds, where the `totalNLL` is the totally negative log-likelihood for whole phrase table.

### Model file
After the training process completed, a file named `model.gz` will be dumped into the directory, and next time the scripts will load parameters from the that file instead of training again.

Note that the file dumped is has two format, GPU trained and CPU trained, and even worse, those tow format are **not compatible**.

There are several unofficial ways to allow the loading.

### Rescoring
The rescoring process also takes a long time, a pre-trained phrase table is attached as `tm.rescore`. Rescoring is simply based on adding log-likelihood.

[1]: http://arxiv.org/pdf/1406.1078.pdf
