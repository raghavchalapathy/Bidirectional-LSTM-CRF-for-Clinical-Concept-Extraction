# Bidirectional-LSTM-CRF-for-Clinical-Concept-Extraction

Extraction of concepts present in patient clinical records is an essential step in clinical research.
The 2010 i2b2/VA Workshop on Natural Language Processing Challenges for clinical records
presented concept extraction (CE) task, with aim to identify concepts (such as treatments, tests,
problems) and classify them into predefined categories. State-of-the-art CE approaches heavily
rely on hand crafted features and domain specific resources which are hard to collect and tune.
For this reason, this paper employs bidirectional LSTM with CRF decoding initialized with general
purpose off-the-shelf word embeddings for CE. The experimental results achieved on 2010
i2b2/VA reference standard corpora using bidirectional LSTM CRF ranks closely with top ranked
systems.

This repository contains the code and  sample data (i2b2-2010) data sets used in the accepted paper "Bidirectional LSTM-CRF for Clinical Concept Extraction"  at Clinical Natural Language Processing Workshop at COLING 2016 Osaka, Japan. December 11, 2016
## If you use this code  in your scientific publications  kindly cite the papers below .
["Bidirectional LSTM-CRF for Clinical Concept Extraction"  at Clinical Natural Language Processing Workshop at COLING 2016 Osaka, Japan. December 11, 2016](https://arxiv.org/abs/1609.07585)

The code utilizes GloVe and Word2Vec pre-trained embeddings file to obtain vector representations.


##If you plan to use the  data(i2b2-2010) 
Please follow the instructions ["here"](https://www.i2b2.org/NLP/DataSets/Agreement.php)


## Initial setup

To use the dnr, you need Python 2.7, with Numpy and Theano installed.


## Using Code for Clinical Concept Extraction

The fastest way to use the dnr  is to use one of the pretrained models:

```
.tagger/conceptExtractor.py --model models/english/ --input input.txt --output output.txt
```

The input file should contain one sentence by line, and they have to be tokenized.
Otherwise, the concept extraction  will perform poorly.


## Train a model

To train your own model, you need to use the train.py script and provide the location of the training,
development and testing set:

```
./train.py --train train.txt --dev dev.txt --test test.txt
```

The training script will automatically give a name to the model and store it in ./models/
There are many parameters you can tune (CRF, dropout rate, embedding dimension, LSTM hidden layer size, etc).
To see all parameters, simply run:

```
./train.py --help
```

Input files for the training script have to follow the same format than the CoNLL2003 sharing task:
each word has to be on a separate line, and there must be an empty line after each sentence.
 A line must contain at least 2 columns, the first one being the word itself, the last one being the named entity.
 It does not matter if there are extra columns that contain tags or chunks in between.
 Tags have to be given in the IOB format (it can be IOB1 or IOB2).


## Train a model in a loop using the Hyper parameters

To train your own model, you need to use the train_loop.py script and provide the location of the training,
development and testing set: if not specified the default locations of ./dnr/data/conll2003/ would be choosen

```
./train_loop.py --train train.txt --dev dev.txt --test test.txt
```

The training script will automatically give a name to the model and store it in ./models/
There are many parameters you can tune (CRF, dropout rate, embedding dimension, LSTM hidden layer size, etc).
To see all parameters, simply run:




