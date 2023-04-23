# Learning unfolded networks with a cyclic group stucture
Official code repository for an under submission paper on learning group actions in convolutional settings.

It consists of an unfolded architecture for learning networks whose layer weights have a cyclic group structure. It is an extension of our previous work, "[Learning unfolded networks with a cyclic group stucture](https://manosth.github.io/files/papers/TheodosisBa_UnfoldedCyclical_NeurReps22.pdf)", an extended abstract accepted at the NeurIPS Workshop for [Symmetry and Geometry in Neural Representations](neurreps.org), but we now are able to learn the group structure.

The data are expected to be (or are downloaded) in the user's home folder and to train type `python3 train.py`. The default architecture is an unfolded network with 4 layers where each layer has 5 base filters and 5 group actions are learned to generate 4 total elements per group. . Trianing should take about 10 mins to run on a GeForce 1080 and reaches ~65% accuracy on CIFAR10.
