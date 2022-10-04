import pickle
import spacy
assert spacy.__version__ == "3.2.2", spacy.__version__
import argparse
from tqdm import tqdm

def pr_at_k(dataset_lists, result_lists, k=10, avg='macro', pct=True):
  if avg == 'micro':
    raise NotImplementedError()
  else:
    numerator = 0
    p_denominator = 0
    r_denominator = 0
    rs_denominator = 0
    # dataset_lists: N, (gold, score)
    # result_lists: N, (cand, score)
    for d, r in zip(dataset_lists, result_lists):
      d = set([s for s, _ in d])
      r = [s for s, _ in r]
      r = r[:k]
      numerator += sum([int(s in d) for s in r]) #matched words
      p_denominator += len(r)
      r_denominator += min(len(d), k)
      rs_denominator += len(d)
    p = numerator / p_denominator
    r = numerator / r_denominator
    f = (2 * p * r) / (p + r)
    # rs = numerator / rs_denominator
    # fs = (2 * p * rs) / (p + rs)
    mul = 100 if pct else 1
    return {
      f'f@{k}': f * mul,
      f'p@{k}': p * mul,
      f'r@{k}': r * mul
    }

def get_lemma2score(dict_tmp, pos):
    phrase_lemma2score = {}
    sorted_dict = [(w, v) for w, v in sorted(dict_tmp.items(), key=lambda item: -1 * item[1])]
    for i in range(len(sorted_dict)):
        w = sorted_dict[i][0]
        score = sorted_dict[i][1]
        w_token = nlp(str(w)) 
        if len(w.split()) == 1:
            assert len(w_token) == 1
        w_token = w_token[0]
        w_token.pos_ = pos
        phrase_lemma = nlp.pipeline[5][1].lemmatize(w_token)
        assert len(phrase_lemma) == 1
        phrase_lemma = phrase_lemma[0]
        if phrase_lemma not in phrase_lemma2score:
            phrase_lemma2score[phrase_lemma] = score
        else:
            phrase_lemma2score[phrase_lemma] = max(phrase_lemma2score[phrase_lemma],score)
        if len(phrase_lemma2score) == 15:
            break
    phrase_lemma2score = [(k,v) for k, v in sorted(phrase_lemma2score.items(), key=lambda item: -1 * item[1])]
    return phrase_lemma2score

parser = argparse.ArgumentParser()
parser.add_argument(
    '-i',
    help='model')
parser.add_argument(
    '-gold',
    help='model')

opt = parser.parse_args()
nlp = spacy.load('it_core_news_sm')

sentfile = "Italian_masked_sent.txt"
goldfile = "Italian_gold.pkl"
tgt_lemma_pos_list = "tgt_lemma_pos_list.txt"

with open(opt.i, 'rb') as f:
    candidate_scores = pickle.load(f)

tgt_lemma_list = []
tgt_pos_list = []

for line in open(tgt_lemma_pos_list, errors='ignore'):
    line = line.rstrip('\n').split('.') #e.g. "sostanza", "n"
    assert len(line)==2
    w = line[0]
    tgt_pos_list.append(line[1]) # "n"
    if w[-1] == "'":
        w = w[:-1]
    tgt_lemma_list.append(w) #sostanza (lemmatised target word)

nonempty_idx = []
with open(goldfile, 'rb') as f: #gold substitutes are pre-lemmatised
    gold = pickle.load(f)
    gold_dict = []
    for i in range(len(gold)):
        words = gold[i]
        if len(words):
            nonempty_idx.append(i)
        gold_dict.append({(w,0) for w in words})

lines = []
for line in open(sentfile, errors='ignore'):
    lines.append(line.rstrip('\n'))

tgt_lemma_spacy_list = []
convert_dict = {"a":"ADJ","v":"VERB","n":"NOUN","r":"ADV"}

for i in range(len(lines)):
    tgt_phrase = lines[i].split("\t")[0]
    if tgt_phrase[-1] == "'":
        tgt_phrase = tgt_phrase[:-1]
    pos = convert_dict[tgt_pos_list[i]]
    tgt_lemma = get_lemma2score({tgt_phrase: 0}, pos)[0][0]
    tgt_lemma_spacy_list.append(tgt_lemma) #tgt word lemma obtained by spacy

assert len(candidate_scores)==len(tgt_lemma_spacy_list)
assert len(tgt_lemma_list)==len(tgt_lemma_spacy_list)
assert len(gold_dict)==len(tgt_lemma_spacy_list)

phrase_lemma2score_list = []
for i in tqdm(range(len(candidate_scores))):#lemmatise predicted candidates
    dict_tmp = candidate_scores[i]
    tgt_phrase = lines[i].split("\t")[0]
    pos = convert_dict[tgt_pos_list[i]]
    phrase_lemma2score = get_lemma2score(dict_tmp, pos)
    phrase_lemma2score_new = []
    for kkk in range(len(phrase_lemma2score)):
        w, val = phrase_lemma2score[kkk]
        if tgt_lemma_spacy_list[i] != w and tgt_lemma_list[i] != w: #omit tgt lemmas
            phrase_lemma2score_new.append((w, val))
    # if len(phrase_lemma2score)==15 :
    #     assert len(phrase_lemma2score_new)>=10
    phrase_lemma2score_list.append(phrase_lemma2score_new)

# Output files for oot and best evaluation
lines = []
for line in open(opt.gold, errors='ignore'):
    line = line.rstrip("\n").split(" :: ")
    lines.append(line[0])

with open(opt.i.split(".pkl")[0] + "_candidates-oot.txt", "w") as f:
    for i in range(len(phrase_lemma2score_list)):
        phrase_list = phrase_lemma2score_list[i][:10] #top 10
        line = lines[i]+ " ::: "
        for j in range(len(phrase_list)):
            w, v=  phrase_list[j]
            line+=w
            line+=";"
        f.write(line+"\n")

with open(opt.i.split(".pkl")[0]+"_candidates-best.txt", "w") as f:
    for i in range(len(phrase_lemma2score_list)):
        phrase_list = phrase_lemma2score_list[i][:10]
        line = lines[i]+ " :: "
        for j in range(1): #top 1
            w, v=  phrase_list[j]
            line+=w
            line+=";"
        f.write(line+"\n")

# calculate F scores
ref = [gold_dict[k] for k in nonempty_idx]  #gold substitutes are pre-lemmatised
sys = [phrase_lemma2score_list[k] for k in nonempty_idx]  #gold substitutes are pre-lemmatised


# with open(opt.i+ "_pred.pkl", 'wb') as f:
#     pickle.dump(sys, f)

val = pr_at_k(ref, sys) #calculate F scores
print(val)

with open(opt.i.split(".pkl")[0] + "_scores.txt", 'wb') as f:
    pickle.dump(val, f)
