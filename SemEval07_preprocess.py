#Download data from "All Gold Standard and Scoring Data" at http://www.dianamccarthy.co.uk/task10index.html

import pickle
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '-folder',
    help='task10data')

opt = parser.parse_args()
sid2cand = {} # sent_id -> candidates
word2cand = {} # e.g. bright.a -> candidates

for file in [opt.folder+"/trial/gold.trial", opt.folder+"/scoring/gold"]:
    for line in open(file, errors='ignore'):
        #bright.a 1 :: intelligent 3;clever 3;smart 1;
        line = line.rstrip("\n").split(" :: ")
        w_and_id = line[0].split() #bright.a 1
        s_id = int(w_and_id[1]) #1; 1~2010
        if len(w_and_id[0].split(".")) != 2:
            assert len(w_and_id[0].split(".")) == 3 #bar.n.v
            w_and_id[0] = ".".join(w_and_id[0].split(".")[:2]) #bar.n
        cand = line[1].split(";") #intelligent 3;clever 3;smart 1;
        assert cand[-1] == ""
        w_list = []
        for w_val in cand[:-1]:
            w_val = w_val.lstrip(" ").rstrip(" ").split(" ")
            # intelligent 3
            assert len(w_val)>=2
            val = int(w_val[-1]) #score
            w = " ".join(w_val[:-1]) #phrase
            w = w.rstrip().lstrip()
            assert len(w.split(" (")) == 1 #no bracket
            if len(w.lstrip().rstrip().split()) ==1 and len(w.lstrip().rstrip().split("-"))==1:
                # if w is NOT MWE
                w_list.append(w)
        sid2cand[s_id] = w_list
        if w_and_id[0] not in word2cand:
            word2cand[w_and_id[0]] = set(w_list)
        else:
            word2cand[w_and_id[0]].update(w_list)

tgt_lemma_orig_list = []
tgt_lemma_list = []
lines_masked = []
sent_id_List = []
# ids_wo_gold = []
candidates = []
flag = False
for file in [opt.folder+"/trial/lexsub_trial.xml",opt.folder+"/test/lexsub_test.xml"]:
    for line in open(file, errors='ignore'):
        line = line.rstrip("\n").lstrip()
        if line.startswith('<instance id="'):#<instance id="11"
            line = line.split('<instance id="')[1]
            line = line.split('">')[0]
            sent_id_List.append(int(line)) #11
        elif line.startswith('<lexelt item="'):
            line = line.split('<lexelt item="')[1]
            line = line.split('">')[0] # bright.a
            tgt_lemma_list.append(line)
        if flag or line.startswith('<context>'):#
            if not flag:
                line = line.split('<context>')[1]
            if line == "":
                flag = True
                continue
            assert line.endswith('</context>')
            line = line.split('</context>')[0].rstrip(" ")
            words = []
            tgt = None
            lemma_tmp = tgt_lemma_list[-1] #bright.a
            tgt_lemma_orig_list.append(lemma_tmp)
            if len(lemma_tmp.split(".")) == 3: #bar.n.v
                lemma_tmp = ".".join(lemma_tmp.split(".")[:-1]) #bar.n
            assert len(lemma_tmp.split(".")) == 2
            cand_tmp = {}
            for w in list(word2cand[lemma_tmp]): #extact candidates for lemma_tmp
                cand_tmp[w] = 0#{word:0}
            candidates.append(cand_tmp)
            for w in line.split(" "):
                if w.startswith("<head>") and w.endswith("</head>"):
                    tgt = w.split("<head>")[1].split("</head>")[0]
                    words.append("<mask>")
                else:
                    words.append(w)
            assert tgt is not None
            lines_masked.append(tgt + "\t" + " ".join(words) )
            if sent_id_List[-1] not in sid2cand:  # if there are no candidates
                pass
                # print("**No gold labels**")
                # print(tgt_lemma_orig_list[-1] + " " + str(sent_id_List[-1]))
                # ids_wo_gold.append(tgt_lemma_orig_list[-1] + " " + str(sent_id_List[-1]))
            flag = False


with open("SemEval07_masked_sent.txt", "w") as f:
    for i in range(len(lines_masked)):
        f.write(lines_masked[i]+"\n")

with open("SemEval07_candidates.pkl", 'wb') as f:
    pickle.dump(candidates, f, pickle.HIGHEST_PROTOCOL)

with open("SemEval07_tgtids.txt", 'w') as f:
    for i in range(len(sent_id_List)):
        f.write(tgt_lemma_orig_list[i]+" "+str(sent_id_List[i]) + "\n")


