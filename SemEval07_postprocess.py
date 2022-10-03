# python jcs/evaluation/lst/lst_gap.py datasets/lst_all.gold out.txt result.txt no-mwe
# tail -n 1 result.txt

import pickle
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '-file')
parser.add_argument(
    '-out')
opt = parser.parse_args()

#ids with no gold labels
ids_wo_gold = ['about.r 567', 'straight.a 773', 'time.n 794', 'yard.n 804', 'easy.a 1298', 'lead.n 1886', 'right.r 1937']

word_tgtid = []
for line in open("SemEval07_tgtids.txt", errors='ignore'):
    # bright.a 8
    word_tgtid.append(line.rstrip('\n'))

with open(opt.file, 'rb') as f:
    results = pickle.load(f)

assert len(results) == 2010

with open(opt.out, 'w') as f:
    for i in range(len(results)):
        out = ["RANKED", word_tgtid[i]] #bright.a 8
        if word_tgtid[i] in ids_wo_gold: #if there is no gold candidate
            f.write("\t".join(out) + "\n")
            continue
        w2score_dict = results[i]
        topwords = [k for k, v in sorted(w2score_dict.items(), key=lambda item: -1 * item[1])]
        for w in topwords:
            out.append(w + " " + str(w2score_dict[w]))
        f.write("\t".join(out) + "\n")
