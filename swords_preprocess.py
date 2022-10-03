import argparse
import re
import string
import gzip
import json
import pickle
parser = argparse.ArgumentParser()
parser.add_argument(
    '-swords_path',
    help='model')
parser.add_argument(
    '-data_split',
    help='model')

opt = parser.parse_args()
substitute_cands_list = []

f_sent  = open("swords_masked_sent_"+opt.data_split+".txt", "w")
with gzip.open(opt.swords_path + "/assets/parsed/swords-v1.1_"+opt.data_split+".json.gz", 'r') as f:
    swords = json.load(f)

x = 0
punct = string.punctuation+"“"+"”"+"-"
from collections import defaultdict
tid_to_sids = defaultdict(list)
for sid, substitute in swords['substitutes'].items():
  tid_to_sids[substitute['target_id']].append(sid)
tgt_word2sent = {}
tgt_word2idx = {}
indices = []
for tid, target in swords['targets'].items():
    x+=1
    context = swords['contexts'][target['context_id']]
    labels = [swords['substitute_labels'][sid] for sid in tid_to_sids[tid]]
    scores = [l.count('TRUE') / len(l) for l in labels]
    text = context['context']
    substitutes = [swords['substitutes'][sid] for sid in tid_to_sids[tid]]
    substitute_cands = [substitute['substitute'] for substitute in substitutes]
    substitute_cands_list.append({w:0 for w in substitute_cands})
    labels = [swords['substitute_labels'][sid] for sid in tid_to_sids[tid]]
    scores = [l.count('TRUE') / len(l) for l in labels]
    POS = target['pos']
    ####normalise the original text####
    text_clean = text.replace("’s","'s")
    text_words = text_clean.split()
    text_clean = " ".join(text_words)
    target_word = target['target']
    if text_clean== "“Let's not get fucked up tonight, though, okay, Rache? We don’t – ” “Sure, babe.":
        # bug fix in the dev split
        assert target_word == "do"
        target_word = "don’t"
    words_last = text[target['offset']:] #text after the target word
    words_last_clean = words_last.replace("’s", "'s")
    words_last_clean = words_last_clean.split()
    hit = False
    re_pattern = re.compile(re.escape(target_word))
    sent_out = text_words.copy()
    sent_out_masked = text_words.copy()
    for i, w in enumerate(text_words):
        found = re_pattern.search(w)
        if found: #if target word is found, w/w.o punctuations
            init, end = found.span()
            apos_s = w[end:] == "'s"
            hyphen_after = (end<len(w) and w[end] == "-")
            hyphen_before = w[init-1] == "-"
            punct_before = init !=0 and all([c in punct for c in list(w[:init])])
            punct_before = punct_before or hyphen_before
            punct_after =  end != len(w) and all([c in punct for c in list(w[end:])])
            punct_after = punct_after or apos_s or hyphen_after
            assert w[init:end] == target_word
            if (init == 0  or punct_before) and (end == len(w) or punct_after):
                if len(text_words[i:]) != len(words_last_clean):
                    # when the target word appears more than once, and the current match is not specified target word
                    pass
                else:
                    if init == 0 and end == len(w):
                        assert sent_out_masked[i] == target_word
                        sent_out_masked[i] = "<mask>"
                    else:
                        if punct_before and punct_after:
                            sent_out_masked[i] = w[:init] + "<mask>" + w[end:]
                        else:
                            if punct_before:
                                sent_out_masked[i] = w[:init] + "<mask>"
                            if punct_after:
                                sent_out_masked[i] = "<mask>" + w[end:]

                    for j in range(1,len(words_last_clean)):
                        assert text_words[i+j] == words_last_clean[j]
                    hit = True
                    break
    assert hit, text_clean+"||"+target_word

    sent_out_masked = " ".join(sent_out_masked)
    f_sent.write(target_word + "\t" + sent_out_masked + "\n")
    if target_word not in tgt_word2sent:
        tgt_word2sent[target_word] = [text_clean]
    else:
        tgt_word2sent[target_word].append(text_clean)

f_sent.close()
with open("swords_substitute_cands_list_" + opt.data_split + ".pkl", 'wb') as f:
    pickle.dump(substitute_cands_list, f, pickle.HIGHEST_PROTOCOL)


# with open("swords_sent_"+opt.data_split+".pkl", 'wb') as f:
#     pickle.dump(tgt_word2sent, f, pickle.HIGHEST_PROTOCOL)
#
# with open("swords_all_candidate_words."+opt.data_split+".txt", "w") as f:
#     for w in list(all_candidates):
#         if len(w.split())==1:
#             f.write(w + "\n")
