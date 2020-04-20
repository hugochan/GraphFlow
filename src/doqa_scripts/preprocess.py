"""
    This file takes a QuAC data file as input and generates the input files for training a conversational reading comprehension model.
"""


import argparse
import json
import re
import time
import string
from collections import Counter
from pycorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://localhost:9000')


def _str(s):
    """ Convert PTB tokens to normal tokens """
    if (s.lower() == '-lrb-'):
        s = '('
    elif (s.lower() == '-rrb-'):
        s = ')'
    elif (s.lower() == '-lsb-'):
        s = '['
    elif (s.lower() == '-rsb-'):
        s = ']'
    elif (s.lower() == '-lcb-'):
        s = '{'
    elif (s.lower() == '-rcb-'):
        s = '}'
    return s


def process(text):
    # text = re.sub('[%s]' % re.escape('/'), ' / ', text)
    # text = re.sub('[%s]' % re.escape('-'), ' - ', text)
    # text = re.sub('[%s]' % re.escape('.'), ' . ', text)
    paragraph = nlp.annotate(text, properties={
                             'annotators': 'tokenize, ssplit, pos, ner',
                             'outputFormat': 'json',
                             'ssplit.newlineIsSentenceBreak': 'two'})

    output = {'word': [],
              # 'lemma': [],
              'pos': [],
              'ner': [],
              'offsets': []}

    for sent in paragraph['sentences']:
        for token in sent['tokens']:
            output['word'].append(_str(token['word']))
            # output['lemma'].append(_str(token['lemma']))
            output['pos'].append(token['pos'])
            output['ner'].append(token['ner'])
            output['offsets'].append((token['characterOffsetBegin'], token['characterOffsetEnd']))
    return output


def normalize_answer(s):
    """Lower text and remove punctuation, storys and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def find_span_with_gt(context, offsets, ground_truth):
    best_f1 = 0.0
    best_span = (len(offsets) - 1, len(offsets) - 1)
    gt = normalize_answer(ground_truth).split()

    ls = [i for i in range(len(offsets))
          if context[offsets[i][0]:offsets[i][1]].lower() in gt]

    for i in range(len(ls)):
        for j in range(i, len(ls)):
            pred = normalize_answer(context[offsets[ls[i]][0]: offsets[ls[j]][1]]).split()
            common = Counter(pred) & Counter(gt)
            num_same = sum(common.values())
            if num_same > 0:
                precision = 1.0 * num_same / len(pred)
                recall = 1.0 * num_same / len(gt)
                f1 = (2 * precision * recall) / (precision + recall)
                if f1 > best_f1:
                    best_f1 = f1
                    best_span = (ls[i], ls[j])
    return best_span


def find_span(offsets, start, end):
    start_index = end_index = -1
    for i, offset in enumerate(offsets):
        if (start_index < 0) or (start >= offset[0]):
            start_index = i
        if (end_index < 0) and (end <= offset[1]):
            end_index = i
    return (start_index, end_index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', '-d', type=str, required=True)
    parser.add_argument('--output_file', '-o', type=str, required=True)
    args = parser.parse_args()

    with open(args.data_file, 'r') as f:
        dataset = json.load(f)

    data = []
    start_time = time.time()
    for i, datum in enumerate(dataset['data']):
        if i % 10 == 0:
            print('processing %d / %d (used_time = %.2fs)...' %
                  (i, len(dataset['data']), time.time() - start_time))
        datum = datum['paragraphs'][0]
        context_str = datum['context']
        _datum = {'context': context_str,
                  'id': datum['id']}
        _datum['annotated_context'] = process(context_str)
        _datum['qas'] = []

        for qa in datum['qas']:
            answers = qa['orig_answer']
            answer = answers['text']
            answer_start = answers['answer_start']
            answer_end = answers['answer_start'] + len(answers['text'])
            if answer == 'CANNOTANSWER':
                answer_start, answer_end = -1, -1

            ans_ls = []
            for ans in qa['answers']:
                ans_ls.append(ans['text'])

            _qas = {'turn_id': qa['id'],
                    'followup': qa['followup'],
                    'yesno': qa['yesno'],
                    'question': qa['question'],
                    'answer': answer,
                    'additional_answers': ans_ls}

            _qas['annotated_question'] = process(qa['question'])
            _qas['annotated_answer'] = process(answer)
            _qas['answer_span_start'] = answer_start
            _qas['answer_span_end'] = answer_end

            start = answer_start
            end = answer_end
            chosen_text = _datum['context'][start: end].lower()
            while len(chosen_text) > 0 and chosen_text[0] in string.whitespace:
                chosen_text = chosen_text[1:]
                start += 1
            while len(chosen_text) > 0 and chosen_text[-1] in string.whitespace:
                chosen_text = chosen_text[:-1]
                end -= 1
            input_text = _qas['answer'].strip().lower()
            if input_text in chosen_text:
                i = chosen_text.find(input_text)
                _qas['answer_span'] = find_span(_datum['annotated_context']['offsets'],
                                                start + i, start + i + len(input_text))
            else:
                _qas['answer_span'] = find_span_with_gt(_datum['context'],
                                                        _datum['annotated_context']['offsets'], input_text)
            _datum['qas'].append(_qas)
        data.append(_datum)

    dataset['data'] = data
    with open(args.output_file, 'w') as output_file:
        json.dump(dataset, output_file, sort_keys=True, indent=4)
