# NeuralCorefRes
Coreference resolution

## Usage
### StanfordCoreNLP
Download the Stanford's CoreNLP [utility here](https://stanfordnlp.github.io/CoreNLP/download.html). Unzip the file and move to a data directory by running the following commands:

`touch data/ && mv path/to/download/location/stanford-corenlp-full-2018-10-05 data/`

TO start the server, run the following:

`java -cp "./stanford-corenlp-full-2018-10-05/*" -mx4g edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000`

and begin using the utility.
