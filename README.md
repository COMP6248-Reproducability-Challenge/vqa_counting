# vqa_counting
### Deep Learning Coursework: The COMP6248 Reproducibility Challenge

## Introduction
For this assignment, we did some experiments on the paper of  [Learning to Count Objects in Natural Images for Visual Question Answering](https://openreview.net/forum?id=B12Js_yRb).
We modify the author's code so that we can do more experiments.

## Experiments
 The additional experiments are as followed:
1. Change the number of weights of piecewise linear activation function to 32 instead of 16 per- formed in the paper.
2. Increase the number of targets to 30 rather than 10 in the paper.
3. Use random length of targets rather than the invariant length of data which is the same as realistic objects.
4. Add a visualize tool to track the calculation of the graph.
5. Change the confidence value to 0.1 rather than 0.5 performed in the paper. 6. Obtain a loss curve by changing epochs.1

## Usage
The main code implemented by us is `visualisation.py`.
For the experiment, we modify the parameter of `train.py` every time.
