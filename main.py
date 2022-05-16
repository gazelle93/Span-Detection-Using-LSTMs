import argparse
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from text_processing import get_target
import my_onehot
from lstms import SpanDetectionLSTM

def main(args):
    cur_text = "However, he also doubts that a hacker would have much interest in the blood pressure readings you're sending to your doctor because if you are in his shoes, you'd find it difficult to make profit off that data."
    ant_span_text = "if you are in his shoes"
    con_span_text = "you'd find it difficult to make profit off that data"

    
    # One-hot Encoding
    sklearn_onehotencoder = my_onehot.build_onehot_encoding_model(args.unk_ignore)
    token2idx_dict, _ = my_onehot.init_token2idx(cur_text, args.nlp_pipeline)
    sklearn_onehotencoder.fit([[t] for t in token2idx_dict])
    tks = my_onehot.get_tokens(cur_text, args.nlp_pipeline)

    embeddings = my_onehot.onehot_encoding(sklearn_onehotencoder, tks)
    target, label_dict, reverse_label_dict = get_target(cur_text, ant_span_text, con_span_text, args.nlp_pipeline)

    input_dim = embeddings.size()[1]
    output_dim = len(set(target))

    if args.lstm.lower() == "lstm":
        model = SpanDetectionLSTM(input_dim=input_dim, output_dim=output_dim, num_layers=args.num_layers, dropout_rate=args.dropout_rate)
    elif args.lstm.lower() == "bilstm":
        model = SpanDetectionLSTM(input_dim=input_dim, output_dim=output_dim, num_layers=args.num_layers, dropout_rate=args.dropout_rate, is_bilstm=True)


    # Training
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    model.zero_grad()

    for _ in tqdm(range(args.epochs)):
        optimizer.zero_grad()

        output = model(embeddings)

        loss = loss_function(output, torch.tensor(target))
        loss.backward()
        optimizer.step()

    print(loss)
    pred = [label_dict[x.item()] for x in torch.argmax(model(embeddings), dim=1)]

    print("Antecedent spans")
    print("True Antecedent spans: {}".format([tks[idx] for idx, x in enumerate(target) if label_dict[x] == "A"]))
    print("Predicted Antecedent spans: {}".format([tks[idx] for idx, x in enumerate(pred) if x == "A"]))

    print("\nConsequent spans")
    print("True Consequent spans: {}".format([tks[idx] for idx, x in enumerate(target) if label_dict[x] == "C"]))
    print("Predicted Consequent spans: {}".format([tks[idx] for idx, x in enumerate(pred) if x == "C"]))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--nlp_pipeline", default="spacy", type=str, help="NLP preprocessing pipeline.")
    parser.add_argument("--unk_ignore", default=True, help="Ignore unknown tokens.")
    parser.add_argument("--num_layers", default=1, type=int, help="The number of lstm/bilstm layers.")
    parser.add_argument("--lstm", default="lstm", type=str, help="Type of lstm layer. (lstm, bilstm)")
    parser.add_argument("--epochs", default=100, type=int, help="The number of epochs for training.")
    parser.add_argument("--learning_rate", default=1e-2, type=float, help="Learning rate.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout rate.")
    args = parser.parse_args()

    main(args)
