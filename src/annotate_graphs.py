import json
import argparse
from collections import defaultdict
import networkx as nx
import spacy
from spacy.tokens import Doc


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

nlp = spacy.load('en')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)


def extract_sent_dep_tree(parser, text):
    if len(text) == 0:
        return {'g_features': [], 'g_adj': {}, 'num_edges': 0}

    doc = parser(text)
    g_features = []
    dep_tree = defaultdict(list)
    boundary_nodes = []
    num_edges = 0
    for sent in doc.sents:
        boundary_nodes.append(sent[-1].i)
        for each in sent:
            g_features.append(each.text)
            if each.i != each.head.i: # Not a root
                dep_tree[each.head.i].append({'node': each.i, 'edge': each.dep_})
                num_edges += 1

    for i in range(len(boundary_nodes) - 1):
        # Add connection between neighboring dependency trees
        dep_tree[boundary_nodes[i]].append({'node': boundary_nodes[i] + 1, 'edge': 'neigh'})
        dep_tree[boundary_nodes[i] + 1].append({'node': boundary_nodes[i], 'edge': 'neigh'})
        num_edges += 2

    info = {'g_features': g_features,
            'g_adj': dep_tree,
            'num_edges': num_edges
            }
    return info


def extract_sent_dep_order_tree(parser, text):
    '''Keep both dependency and ordering info'''
    if len(text) == 0:
        return {'g_features': [], 'g_adj': {}, 'num_edges': 0}

    doc = parser(text)
    g_features = []
    dep_tree = defaultdict(list)

    num_edges = 0
    # Add word ordering info
    for i in range(len(doc) - 1):
        dep_tree[i].append({'node': i + 1, 'edge': 'neigh'})
        dep_tree[i + 1].append({'node': i, 'edge': 'neigh'})
        num_edges += 2


    # Add dependency info
    for sent in doc.sents:
        for each in sent:
            g_features.append(each.text)
            if each.i != each.head.i and abs(each.head.i - each.i) != 1: # Not a root
                dep_tree[each.head.i].append({'node': each.i, 'edge': each.dep_})
                num_edges += 1

    info = {'g_features': g_features,
            'g_adj': dep_tree,
            'num_edges': num_edges
            }
    return info

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, type=str, help='path to the input file')
    parser.add_argument('-o', '--output', required=True, type=str, help='path to the output file')
    parser.add_argument('-order', '--word_order', action='store_true', help='whether to include word ordering info')
    args = vars(parser.parse_args())


    extract_graph_func = extract_sent_dep_order_tree if args['word_order'] else extract_sent_dep_tree

    with open(args['input']) as dataset_file:
        dataset = json.load(dataset_file, encoding='utf-8')

        for instance in dataset['data']:
            text = instance['annotated_context']['word']
            graph = extract_graph_func(nlp, ' '.join(text))
            instance['annotated_context']['graph'] = graph

        with open(args['output'], 'w') as out_file:
            json.dump(dataset, out_file)
