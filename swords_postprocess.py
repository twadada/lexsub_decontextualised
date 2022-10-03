import gzip
import json
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    '-i')
parser.add_argument(
    '-swords_path')
parser.add_argument(
    '-data_split',
    default='test')
parser.add_argument(
    '-save_name')
opt = parser.parse_args()


testfile = opt.swords_path+'/assets/parsed/swords-v1.1_'+opt.data_split+'.json.gz'
with gzip.open(testfile, 'r') as f:
  swords = json.load(f)

import pickle

with open( opt.i, 'rb') as f:
    candidate_scores = pickle.load(f)

from collections import defaultdict
tid_to_sids = defaultdict(list)
for sid, substitute in swords['substitutes'].items():
    tid_to_sids[substitute['target_id']].append(sid)

result = {'substitutes_lemmatized': False, 'substitutes': {}}
assert len(swords['targets'].items())== len(candidate_scores),str(len(swords['targets'].items()))+"_"+str(len(candidate_scores))
count = 0
for tid, target in swords['targets'].items():
    context = swords['contexts'][target['context_id']]
    result_tmp = []
    key_val = candidate_scores[count]
    count += 1
    substitutes = [swords['substitutes'][sid] for sid in tid_to_sids[tid]]
    assert len(key_val) == len(substitutes) or len(key_val) == 50
    # labels = [swords['substitute_labels'][sid] for sid in tid_to_sids[tid]]
    # scores = [l.count('TRUE') / len(l) for l in labels]
    # text = context['context']
    sorted_key = [k for k, v in sorted(key_val.items(), key=lambda item: -1 * item[1])]
    c = 0
    substitute_to_max_score = {}
    for j, y in enumerate(sorted_key):
        if target["target"] !=y: #omit the target word
            if y not in substitute_to_max_score:
                substitute_to_max_score[y] = key_val[y]
            else:
                val = substitute_to_max_score[y]
                substitute_to_max_score[y] = max(val, key_val[y])
    result_tmp = []
    for w in substitute_to_max_score.keys():
        result_tmp.append((w, substitute_to_max_score[w]))
    idx = np.argsort([x[1] for x in result_tmp])[::-1] # sorted by scores
    result['substitutes'][tid] = [result_tmp[i] for i in idx]

with open(opt.save_name, 'w') as f:
    f.write(json.dumps(result))


