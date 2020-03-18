# NeuralCorefRes
Coreference resolution using a CNN/LSTM neural network.

Ryan Elliott

## Usage
### Training
Training requires the `data` directory to, at the very least, containg the following paths:

- `data/PreCo_1.0/custom_dps/train_b<n>.json`
- `data/models/word_embeddings/preco-vectors.model`

You can then proceed to run `python3 generate_data.py -t`. Please note this command will not train on the entire dataset. This has been limited to one file to avoid requiring the use of batches. If the program is terminated before it's able to start training, reduce the number of training samples. This can be done by adding an optional flag, `--samples <num>`.

### Predictions
Run `python3 generate_data.py -p "This is a sentence that should be predicted on."`

### StanfordCoreNLP
Download the Stanford CoreNLP and CoreNLP server [utility here](https://stanfordnlp.github.io/CoreNLP/download.html). Unzip the file and move to a data directory by running the following commands:

`touch data/ && mv <path/to/download/location>/stanford-corenlp-full-2018-10-05 data/`

To start the server and begin using the utility, run the following:

`java -cp "./<path/to/data/directory>/stanford-corenlp-full-2018-10-05/*" -mx4g edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000`

Since I understand it can be confusing trying to traverse other's file hiearchies, the following is the output of my running `tree -d -I 'scripts|v4|patterns|sutime|...<directories_i_want_to_hide>'` in the root directory:
```
.
├── LICENSE
├── README.md
├── data
│   ├── conll-2012
│   └── stanford-corenlp-full-2018-10-05
│       ...
├── neuralcorefres
│   ├── feature_extraction
│   │   ├── __init__.py
│   │   ├── gender_classifier.py
│   │   └── stanford_parser.py
│   └── test.py
├── requirements.txt
└── tests
    └── test_feature_extraction.py

8 directories, 53 files
```
Using the above file hierarchy, I run the below command from `./` to start the Stanford CoreNLP server:

`java -cp "./data/stanford-corenlp-full-2018-10-05/*" -mx4g edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000`


## Code Conventions
### Typing
All code in this repo makes use of PEP 484, Python3 type hints. Generally, typing variables within functions is redundant and leads to excessive typing. Therefore, types are typically only explicitly stated in function signatures. However, per any rule in computer science, there are exceptions; if a variable is particularly ambiguous, explicitly stating its types makes sense should be encouraged.
