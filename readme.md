# QA2Sent

This is a tool for converting question-answer pair to a declarative sentence. For example, it converts

```
Q: What is the current series where the new series began in June 2011?
A: 'CB·06·ZZ'
```

to

```
CB·06·ZZ is the current series where the new series began in June 2011.
```

To see how to use this tool, run `python qa2sent.py` and modify its inputs and outputs.

Most of the codes come from (robinjia/adversarial-squad)[https://github.com/robinjia/adversarial-squad).

### Dependencies

```
wget "http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip"
unzip stanford-corenlp-full-2018-10-05.zip
pip install pattern stanza tqdm
```
