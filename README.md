# Overview
- Long Short-Term Memory (LSTM) networks are a type of neural network architecture that has capable of learning from sequential data and is used in sequence prediction problems. This project aims to implement the LSTM and Bidirectional LSTM for span detection from given text.

# Brief description
- text_processing.py
> Output format
> - output: Tokenized result of a given text. (list)
- my_onehot.py
> Output format
> - output: List of tensor of input tokens. (Tensor)
- lstms.py
> Output format
> - output: List of tensor of attention results. (Tensor)


# Prerequisites
- argparse
- torch
- stanza
- spacy
- nltk
- gensim
- tqdm

# Parameters
- nlp_pipeline(str, defaults to "stanza"): NLP preprocessing pipeline.
- unk_ignore(bool, defaults to True): Ignore unseen word or not.
- num_layers(int, defaults to 1): The number of lstm/bilstm layers.
- lstm(str, defaults to "lstm"): Type of lstm layer. (lstm, bilstm)
- epochs(int, defaults to 100): The number of epochs for training.
- learning_rate(float, defaults to 1e-2): Learning rate.
- dropout_rate(float, defaults to 0.1): Dropout rate.

# References
- LSTM: Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.
- Stanza: Qi, P., Zhang, Y., Zhang, Y., Bolton, J., & Manning, C. D. (2020). Stanza: A Python natural language processing toolkit for many human languages. arXiv preprint arXiv:2003.07082.
- Spacy: Matthew Honnibal and Ines Montani. 2017. spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing. To appear (2017).
- NLTK: Bird, Steven, Edward Loper and Ewan Klein (2009). Natural Language Processing with Python. O'Reilly Media Inc.
- Gensim: Rehurek, R., & Sojka, P. (2010). Software framework for topic modelling with large corpora. In In Proceedings of the LREC 2010 workshop on new challenges for NLP frameworks.
- Sample data: Yang, X., Obadinma, S., Zhao, H., Zhang, Q., Matwin, S., & Zhu, X. (2020). SemEval-2020 task 5: Counterfactual recognition. arXiv preprint arXiv:2008.00563.
