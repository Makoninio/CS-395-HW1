#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2021-2022: Homework 3
parser_utils.py: Utilities for training the dependency parser.
"""

import time
import os
import logging
from collections import Counter
from . general_utils import get_minibatches
from parser_transitions import minibatch_parse

from tqdm import tqdm
import torch
import numpy as np

P_PREFIX = '<p>:'
L_PREFIX = '<l>:'
UNK = '<UNK>'
NULL = '<NULL>'
ROOT = '<ROOT>'


class Config(object):
    language = 'english'
    with_punct = True
    unlabeled = True
    lowercase = True
    use_pos = True
    use_dep = True
    use_dep = use_dep and (not unlabeled)
    data_path = './data'
    train_file = 'train.conll'
    dev_file = 'dev.conll'
    test_file = 'test.conll'
    embedding_file = './data/en-cw.txt'


class Parser(object):
    """Contains everything needed for transition-based dependency parsing except for the model"""

    def __init__(self, dataset):
        """Build token dictionaries and define the parser's transition set."""
        # Find the dependency label used for the sentence root in the dataset.
        root_labels = list([l for ex in dataset
                           for (h, l) in zip(ex['head'], ex['label']) if h == 0])
        counter = Counter(root_labels)
        if len(counter) > 1:
            logging.info('Warning: more than one root label')
            logging.info(counter)
        self.root_label = counter.most_common()[0][0]
        deprel = [self.root_label] + list(set([w for ex in dataset
                                               for w in ex['label']
                                               if w != self.root_label]))
        tok2id = {L_PREFIX + l: i for (i, l) in enumerate(deprel)}
        tok2id[L_PREFIX + NULL] = self.L_NULL = len(tok2id)

        config = Config()
        self.unlabeled = config.unlabeled
        self.with_punct = config.with_punct
        self.use_pos = config.use_pos
        self.use_dep = config.use_dep
        self.language = config.language

        if self.unlabeled:
            # This project predicts only arc direction, not dependency names.
            # Class IDs are 0=LEFT-ARC, 1=RIGHT-ARC, and 2=SHIFT.
            trans = ['L', 'R', 'S']
            self.n_deprel = 1
        else:
            trans = ['L-' + l for l in deprel] + ['R-' + l for l in deprel] + ['S']
            self.n_deprel = len(deprel)

        self.n_trans = len(trans)
        self.tran2id = {t: i for (i, t) in enumerate(trans)}
        self.id2tran = {i: t for (i, t) in enumerate(trans)}

        # Give every POS tag an integer ID, plus IDs for unknown, missing, and
        # ROOT positions. Word IDs are added to the same shared token table.
        tok2id.update(build_dict([P_PREFIX + w for ex in dataset for w in ex['pos']],
                                  offset=len(tok2id)))
        tok2id[P_PREFIX + UNK] = self.P_UNK = len(tok2id)
        tok2id[P_PREFIX + NULL] = self.P_NULL = len(tok2id)
        tok2id[P_PREFIX + ROOT] = self.P_ROOT = len(tok2id)

        # Frequent words receive IDs; unseen words later map to UNK.
        tok2id.update(build_dict([w for ex in dataset for w in ex['word']],
                                  offset=len(tok2id)))
        tok2id[UNK] = self.UNK = len(tok2id)
        tok2id[NULL] = self.NULL = len(tok2id)
        tok2id[ROOT] = self.ROOT = len(tok2id)

        self.tok2id = tok2id
        self.id2tok = {v: k for (k, v) in tok2id.items()}

        # Current settings use 18 word + 18 POS features = 36 total inputs.
        self.n_features = 18 + (18 if config.use_pos else 0) + (12 if config.use_dep else 0)
        self.n_tokens = len(tok2id)

    def vectorize(self, examples):
        """Replace words, POS tags, and labels with integer vocabulary IDs."""
        vec_examples = []
        for ex in examples:
            word = [self.ROOT] + [self.tok2id[w] if w in self.tok2id
                                  else self.UNK for w in ex['word']]
            pos = [self.P_ROOT] + [self.tok2id[P_PREFIX + w] if P_PREFIX + w in self.tok2id
                                   else self.P_UNK for w in ex['pos']]
            head = [-1] + ex['head']
            label = [-1] + [self.tok2id[L_PREFIX + w] if L_PREFIX + w in self.tok2id
                            else -1 for w in ex['label']]
            vec_examples.append({'word': word, 'pos': pos,
                                 'head': head, 'label': label})
        return vec_examples

    def extract_features(self, stack, buf, arcs, ex):
        """Describe one parser state using a fixed-length list of token IDs.

        The model receives the top stack items, first buffer items, and selected
        children/grandchildren. Missing positions are filled with NULL IDs.
        """
        if stack[0] == "ROOT":
            # In vectorized examples, position 0 is the artificial ROOT token.
            stack[0] = 0

        def get_lc(k):
            return sorted([arc[1] for arc in arcs if arc[0] == k and arc[1] < k])

        def get_rc(k):
            return sorted([arc[1] for arc in arcs if arc[0] == k and arc[1] > k],
                          reverse=True)

        p_features = []
        l_features = []
        # Start with the top 3 stack words and first 3 buffer words.
        features = [self.NULL] * (3 - len(stack)) + [ex['word'][x] for x in stack[-3:]]
        features += [ex['word'][x] for x in buf[:3]] + [self.NULL] * (3 - len(buf))
        if self.use_pos:
            p_features = [self.P_NULL] * (3 - len(stack)) + [ex['pos'][x] for x in stack[-3:]]
            p_features += [ex['pos'][x] for x in buf[:3]] + [self.P_NULL] * (3 - len(buf))

        for i in range(2):
            if i < len(stack):
                # For each of the top two stack words, include its first/second
                # left and right children plus one outer grandchild on each side.
                k = stack[-i-1]
                lc = get_lc(k)
                rc = get_rc(k)
                llc = get_lc(lc[0]) if len(lc) > 0 else []
                rrc = get_rc(rc[0]) if len(rc) > 0 else []

                features.append(ex['word'][lc[0]] if len(lc) > 0 else self.NULL)
                features.append(ex['word'][rc[0]] if len(rc) > 0 else self.NULL)
                features.append(ex['word'][lc[1]] if len(lc) > 1 else self.NULL)
                features.append(ex['word'][rc[1]] if len(rc) > 1 else self.NULL)
                features.append(ex['word'][llc[0]] if len(llc) > 0 else self.NULL)
                features.append(ex['word'][rrc[0]] if len(rrc) > 0 else self.NULL)

                if self.use_pos:
                    p_features.append(ex['pos'][lc[0]] if len(lc) > 0 else self.P_NULL)
                    p_features.append(ex['pos'][rc[0]] if len(rc) > 0 else self.P_NULL)
                    p_features.append(ex['pos'][lc[1]] if len(lc) > 1 else self.P_NULL)
                    p_features.append(ex['pos'][rc[1]] if len(rc) > 1 else self.P_NULL)
                    p_features.append(ex['pos'][llc[0]] if len(llc) > 0 else self.P_NULL)
                    p_features.append(ex['pos'][rrc[0]] if len(rrc) > 0 else self.P_NULL)

                if self.use_dep:
                    l_features.append(ex['label'][lc[0]] if len(lc) > 0 else self.L_NULL)
                    l_features.append(ex['label'][rc[0]] if len(rc) > 0 else self.L_NULL)
                    l_features.append(ex['label'][lc[1]] if len(lc) > 1 else self.L_NULL)
                    l_features.append(ex['label'][rc[1]] if len(rc) > 1 else self.L_NULL)
                    l_features.append(ex['label'][llc[0]] if len(llc) > 0 else self.L_NULL)
                    l_features.append(ex['label'][rrc[0]] if len(rrc) > 0 else self.L_NULL)
            else:
                features += [self.NULL] * 6
                if self.use_pos:
                    p_features += [self.P_NULL] * 6
                if self.use_dep:
                    l_features += [self.L_NULL] * 6

        features += p_features + l_features
        assert len(features) == self.n_features
        return features

    def get_oracle(self, stack, buf, ex):
        """Return the gold transition for a state using the known training tree.

        The oracle exists only while creating training examples. At evaluation
        time, the neural model must make this choice itself.
        """
        if len(stack) < 2:
            # With fewer than two stack items, an arc is impossible; SHIFT.
            return self.n_trans - 1

        i0 = stack[-1]
        i1 = stack[-2]
        h0 = ex['head'][i0]
        h1 = ex['head'][i1]
        l0 = ex['label'][i0]
        l1 = ex['label'][i1]

        if self.unlabeled:
            # If the top item is the gold head of the item below it, LEFT-ARC.
            if (i1 > 0) and (h1 == i0):
                return 0
            # If the lower item heads the top item, RIGHT-ARC only after all of
            # the top item's children have already left the buffer.
            elif (i1 >= 0) and (h0 == i1) and \
                 (not any([x for x in buf if ex['head'][x] == i0])):
                return 1
            else:
                # Otherwise bring in another word, if one remains.
                return None if len(buf) == 0 else 2
        else:
            if (i1 > 0) and (h1 == i0):
                return l1 if (l1 >= 0) and (l1 < self.n_deprel) else None
            elif (i1 >= 0) and (h0 == i1) and \
                 (not any([x for x in buf if ex['head'][x] == i0])):
                return l0 + self.n_deprel if (l0 >= 0) and (l0 < self.n_deprel) else None
            else:
                return None if len(buf) == 0 else self.n_trans - 1

    def create_instances(self, examples):
        """Use the oracle to turn gold trees into supervised examples.

        Every intermediate parser state becomes one item containing its 36
        features, legal-action mask, and correct transition class.
        """
        all_instances = []
        succ = 0
        for id, ex in enumerate(examples):
            n_words = len(ex['word']) - 1

            # arcs = {(h, t, label)}
            stack = [0]
            buf = [i + 1 for i in range(n_words)]
            arcs = []
            instances = []
            # A projective n-word tree takes n SHIFTs and n arc actions.
            for i in range(n_words * 2):
                gold_t = self.get_oracle(stack, buf, ex)
                if gold_t is None:
                    break
                legal_labels = self.legal_labels(stack, buf)
                assert legal_labels[gold_t] == 1
                instances.append((self.extract_features(stack, buf, arcs, ex),
                                  legal_labels, gold_t))
                # Apply the oracle action so the next example represents the
                # following state along the correct transition sequence.
                if gold_t == self.n_trans - 1:
                    stack.append(buf[0])
                    buf = buf[1:]
                elif gold_t < self.n_deprel:
                    arcs.append((stack[-1], stack[-2], gold_t))
                    stack = stack[:-2] + [stack[-1]]
                else:
                    arcs.append((stack[-2], stack[-1], gold_t - self.n_deprel))
                    stack = stack[:-1]
            else:
                succ += 1
                all_instances += instances

        return all_instances

    def legal_labels(self, stack, buf):
        """Return a 0/1 mask for transitions allowed in the current state."""
        # LEFT-ARC needs more than ROOT plus one word, RIGHT-ARC needs two
        # stack items, and SHIFT needs at least one buffered word.
        labels = ([1] if len(stack) > 2 else [0]) * self.n_deprel
        labels += ([1] if len(stack) >= 2 else [0]) * self.n_deprel
        labels += [1] if len(buf) > 0 else [0]
        return labels

    def parse(self, dataset, eval_batch_size=5000):
        """Parse a dataset with the model and calculate unlabeled attachment score."""
        sentences = []
        sentence_id_to_idx = {}
        for i, example in enumerate(dataset):
            n_words = len(example['word']) - 1
            sentence = [j + 1 for j in range(n_words)]
            sentences.append(sentence)
            sentence_id_to_idx[id(sentence)] = i

        model = ModelWrapper(self, dataset, sentence_id_to_idx)
        dependencies = minibatch_parse(sentences, model, eval_batch_size)

        # UAS is the fraction of words whose predicted head equals the gold head.
        UAS = all_tokens = 0.0
        with tqdm(total=len(dataset)) as prog:
            for i, ex in enumerate(dataset):
                head = [-1] * len(ex['word'])
                for h, t, in dependencies[i]:
                    head[t] = h
                for pred_h, gold_h, gold_l, pos in \
                        zip(head[1:], ex['head'][1:], ex['label'][1:], ex['pos'][1:]):
                        assert self.id2tok[pos].startswith(P_PREFIX)
                        pos_str = self.id2tok[pos][len(P_PREFIX):]
                        if (self.with_punct) or (not punct(self.language, pos_str)):
                            UAS += 1 if pred_h == gold_h else 0
                            all_tokens += 1
                prog.update(i + 1)
        UAS /= all_tokens
        return UAS, dependencies


class ModelWrapper(object):
    """Adapt ParserModel's numeric logits to minibatch_parse's action strings."""
    def __init__(self, parser, dataset, sentence_id_to_idx):
        self.parser = parser
        self.dataset = dataset
        self.sentence_id_to_idx = sentence_id_to_idx

    def predict(self, partial_parses):
        # Convert every stack/buffer state into the model's 36 feature IDs.
        mb_x = [self.parser.extract_features(p.stack, p.buffer, p.dependencies,
                                             self.dataset[self.sentence_id_to_idx[id(p.sentence)]])
                for p in partial_parses]
        mb_x = np.array(mb_x).astype('int32')
        mb_x = torch.from_numpy(mb_x).long()
        # Mask tells us which of LA, RA, and SHIFT are structurally legal.
        mb_l = [self.parser.legal_labels(p.stack, p.buffer) for p in partial_parses]

        # The model returns 3 raw scores per state. detach() removes the tensors
        # from gradient tracking because evaluation does not backpropagate.
        pred = self.parser.model(mb_x)
        pred = pred.detach().numpy()
        # Add a huge bonus to legal actions, then greedily choose the highest
        # legal score for each row. Different rows may choose different actions.
        pred = np.argmax(pred + 10000 * np.array(mb_l).astype('float32'), 1)
        # Convert numeric class IDs back to the strings parse_step expects.
        pred = ["S" if p == 2 else ("LA" if p == 0 else "RA") for p in pred]
        return pred


def read_conll(in_file, lowercase=False, max_example=None):
    """Read CoNLL rows into sentences with words, POS tags, heads, and labels."""
    examples = []
    with open(in_file) as f:
        word, pos, head, label = [], [], [], []
        for line in f.readlines():
            sp = line.strip().split('\t')
            if len(sp) == 10:
                if '-' not in sp[0]:
                    word.append(sp[1].lower() if lowercase else sp[1])
                    pos.append(sp[4])
                    head.append(int(sp[6]))
                    label.append(sp[7])
            elif len(word) > 0:
                examples.append({'word': word, 'pos': pos, 'head': head, 'label': label})
                word, pos, head, label = [], [], [], []
                if (max_example is not None) and (len(examples) == max_example):
                    break
        if len(word) > 0:
            examples.append({'word': word, 'pos': pos, 'head': head, 'label': label})
    return examples


def build_dict(keys, n_max=None, offset=0):
    count = Counter()
    for key in keys:
        count[key] += 1
    ls = count.most_common() if n_max is None \
        else count.most_common(n_max)

    return {w[0]: index + offset for (index, w) in enumerate(ls)}


def punct(language, pos):
    if language == 'english':
        return pos in ["''", ",", ".", ":", "``", "-LRB-", "-RRB-"]
    elif language == 'chinese':
        return pos == 'PU'
    elif language == 'french':
        return pos == 'PUNC'
    elif language == 'german':
        return pos in ["$.", "$,", "$["]
    elif language == 'spanish':
        # http://nlp.stanford.edu/software/spanish-faq.shtml
        return pos in ["f0", "faa", "fat", "fc", "fd", "fe", "fg", "fh",
                       "fia", "fit", "fp", "fpa", "fpt", "fs", "ft",
                       "fx", "fz"]
    elif language == 'universal':
        return pos == 'PUNCT'
    else:
        raise ValueError('language: %s is not supported.' % language)


def minibatches(data, batch_size):
    """Package feature vectors and correct transition IDs into shuffled batches."""
    x = np.array([d[0] for d in data])
    y = np.array([d[2] for d in data])
    # Store labels as one-hot rows; run.py converts them back to class IDs for
    # PyTorch's CrossEntropyLoss.
    one_hot = np.zeros((y.size, 3))
    one_hot[np.arange(y.size), y] = 1
    return get_minibatches([x, one_hot], batch_size)


def load_and_preprocess_data(reduced=True):
    """Load files and produce everything needed for training and evaluation."""
    config = Config()

    print("Loading data...",)
    start = time.time()
    train_set = read_conll(os.path.join(config.data_path, config.train_file),
                           lowercase=config.lowercase)
    dev_set = read_conll(os.path.join(config.data_path, config.dev_file),
                         lowercase=config.lowercase)
    test_set = read_conll(os.path.join(config.data_path, config.test_file),
                          lowercase=config.lowercase)
    if reduced:
        train_set = train_set[:1000]
        dev_set = dev_set[:500]
        test_set = test_set[:500]
    print("took {:.2f} seconds".format(time.time() - start))

    print("Building parser...",)
    start = time.time()
    parser = Parser(train_set)
    print("took {:.2f} seconds".format(time.time() - start))

    print("Loading pretrained embeddings...",)
    start = time.time()
    word_vectors = {}
    for line in open(config.embedding_file).readlines():
        sp = line.strip().split()
        word_vectors[sp[0]] = [float(x) for x in sp[1:]]
    embeddings_matrix = np.asarray(np.random.normal(0, 0.9, (parser.n_tokens, 50)), dtype='float32')

    # Replace random rows with pretrained 50-value vectors when available.
    for token in parser.tok2id:
        i = parser.tok2id[token]
        if token in word_vectors:
            embeddings_matrix[i] = word_vectors[token]
        elif token.lower() in word_vectors:
            embeddings_matrix[i] = word_vectors[token.lower()]
    print("took {:.2f} seconds".format(time.time() - start))

    print("Vectorizing data...",)
    start = time.time()
    train_set = parser.vectorize(train_set)
    dev_set = parser.vectorize(dev_set)
    test_set = parser.vectorize(test_set)
    print("took {:.2f} seconds".format(time.time() - start))

    print("Preprocessing training data...",)
    start = time.time()
    # Only the training set needs oracle-generated state/transition examples.
    train_examples = parser.create_instances(train_set)
    print("took {:.2f} seconds".format(time.time() - start))

    return parser, embeddings_matrix, train_examples, dev_set, test_set,


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    pass
