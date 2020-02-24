# NeuralCorefRes
Coreference resolution

## Usage
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
