import os
import json
import argparse
from tqdm import tqdm


COCO = {
    'adversarial': './data/POPE/coco/coco_pope_adversarial.json',
    'popular': './data/POPE/coco/coco_pope_popular.json',
    'random': './data/POPE/coco/coco_pope_random.json',
}
AOKVQA = {
    'adversarial': './data/POPE/aokvqa/aokvqa_pope_adversarial.json',
    'popular': './data/POPE/aokvqa/aokvqa_pope_popular.json',
    'random': './data/POPE/aokvqa/aokvqa_pope_random.json',
}

parser = argparse.ArgumentParser()
parser.add_argument('--gt-files', type=str, default=None)
parser.add_argument('--model-outputs', type=str, default='')
args = parser.parse_args()

# open ground truth answers
if args.gt_files is None:
    if 'coco' in args.model_outputs:
        data_type = COCO
    else:
        data_type = AOKVQA
    if 'adversarial' in args.model_outputs:
        args.gt_files = data_type['adversarial']
    elif 'popular' in args.model_outputs:
        args.gt_files = data_type['popular']
    else:
        args.gt_files = data_type['random']
print(args.gt_files)
gt_files = [json.loads(q) for q in open(os.path.expanduser(args.gt_files), 'r')]

# open generated answers
gen_files = [json.loads(q) for q in open(os.path.expanduser(args.model_outputs), 'r')]

# index generated answers by question_id -> robust to ordering AND to partial
# runs (e.g. --limit), which positional matching could not handle.
gen_by_id = {g['question_id']: g for g in gen_files}

true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
unknown = 0
yes_answers = 0
evaluated = 0

# compare answers (only questions that were actually generated)
for line in gt_files:
    idx = line['question_id']
    if idx not in gen_by_id:
        continue  # not answered in this run (partial / limited)
    gt_answer = line['label'].lower().strip()
    gen_answer = gen_by_id[idx]['text'].lower().strip()
    evaluated += 1
    # pos = 'yes', neg = 'no'
    if gt_answer == 'yes':
        if 'yes' in gen_answer:
            true_pos += 1
            yes_answers += 1
        else:
            false_neg += 1
    elif gt_answer == 'no':
        if 'no' in gen_answer:
            true_neg += 1
        else:
            yes_answers += 1
            false_pos += 1
    else:
        print(f'Warning: unknown gt_answer: {gt_answer}')
        unknown += 1

total_questions = evaluated
# precision, recall, f1, accuracy, proportion of 'yes' answers (guard /0)
precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) else 0.0
recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
accuracy = (true_pos + true_neg) / total_questions if total_questions else 0.0
yes_proportion = yes_answers / total_questions if total_questions else 0.0

# report results
print(f'Evaluated: {evaluated} / {len(gt_files)} gt questions')
print(f'Accuracy: {accuracy*100:.2f}')
print(f'Precision: {precision*100:.2f}  Recall: {recall*100:.2f}')
print(f'F1: {f1*100:.2f}')
print(f'Yes proportion: {yes_proportion*100:.2f}\n')
