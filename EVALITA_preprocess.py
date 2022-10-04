# download test data at https://www.evalita.it/campaigns/evalita-2009/tasks/lexical-substitution
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '-folder')

opt = parser.parse_args()

s_id = []
w_list_all = []
tgt_list = []
for line in open(opt.folder+"/gold.test", errors='ignore'):
    line = line.rstrip("\n").split(" :: ")
    w_and_id = line[0].split()
    s_id.append(int(w_and_id[1]))
    assert len(w_and_id[0].split("."))==2
    tgt_list.append(w_and_id[0].split(".")[0])
    cand = line[1].split(";")
    assert cand[-1]== ""
    w_list = []
    for w_val in cand[:-1]:
        w_val = w_val.lstrip(" ").rstrip(" ").split(" ")
        assert len(w_val)>=2
        val = int(w_val[-1])
        w = " ".join(w_val[:-1]).rstrip(" ").lstrip(" ")
        if len(w.split(" ("))==2:
            w = w.split(" ")[0] #omit words in brackets
        else:
            assert len(w.split(" ("))==1
        if w[-1]=="'":
            w = w[:-1] #omit '
        w_list.append(w)
    w_list_all.append(w_list)

import pickle
with open("Italian_gold.pkl", 'wb') as f:
    pickle.dump(w_list_all, f, pickle.HIGHEST_PROTOCOL)


lines = []
tgts = []
flag = False
count = 0
import spacy
assert spacy.__version__ == "3.2.2", spacy.__version__
nlp = spacy.load("it_core_news_sm")
idx = []
# un_idx= []
# una_idx= []
# il_idx= []
# la_idx= []
# le_idx= []
# i_idx= []
tgt_lemma_pos_list = []
for line in open(opt.folder+"/lexsub_test.xml", errors='ignore'):
    line = line.rstrip("\n")
    if line.startswith('<lexelt item="'):
        line = line.split('<lexelt item="')[1]
        tgt_lemma_pos = line.split('">')[0] #e.g. "sostanza.n"
    if line.startswith('<instance id="'):#e.g. <instance id="2311">
        line = line.split('<instance id="')[1]
        line = line.split('">')[0]
        if count==len(s_id): #discard sentences without gold candidates
            break
        assert int(line) == s_id[count] #same id
        count += 1
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
        if len(line.split())>300: #If the target context is long
            line = nlp(line) #split into sentences
            line = list(line.sents)
            for kk, sent in enumerate(line):
                if "<head>" in sent.text: #include preceeding and following sents
                    line = line[kk-1].text + " "+ line[kk].text+ " " + line[kk+1].text
        for w in line.split(" "):
            if w.startswith("<head>") and w.endswith("</head>"):
                #e.g. <head>sostanza</head>
                tgt = w.split("<head>")[1].split("</head>")[0]
                # if tgt != tgt_list[count-1]:
                #     print(tgt+"@"+tgt_list[count-1])
                assert w.split("</head>")[-1]== ""
                words.append("<mask>")
            elif w.startswith("<head>"):
                #e.g. <head>auto</head>mobilistica
                tgt_words = w.split("<head>")[1].split("</head>")
                assert len(tgt_words)==2, w + "@"+line
                tgt = tgt_words[0]
                words.append("<mask>")
                words.append(tgt_words[1]) #mobilistica
            elif w.endswith("</head>"):
                #something<head>tgt</head>
                tgt_words = w.split("</head>")[0].split("<head>")
                assert len(tgt_words)==2
                tgt = tgt_words[1]
                words.append(tgt_words[0]) #something
                words.append("<mask>")
            elif w=="di<head>chiara</head>va": #one exception
                tgt = "chiara"
                words.append("di")
                words.append("<mask>")
                words.append("va")
            elif w=="tras<head>cura</head>tezze": #one exception
                tgt = "cura"
                words.append("tras")
                words.append("<mask>")
                words.append("tezze")
            elif w=="assi<head>cura</head>rsi": #one exception
                tgt = "cura"
                words.append("assi")
                words.append("<mask>")
                words.append("rsi")
            else:
                words.append(w)
        assert tgt is not None, words
        if tgt[-1]=="'":
            tgt = tgt[:-1] #remove apos
        lines.append(tgt + "\t" +" ".join(words) )
        tgt_lemma_pos_list.append(tgt_lemma_pos)
        flag = False

with open("Italian_masked_sent.txt", "w") as f:
    for i in range(len(lines)):
        f.write(lines[i]+"\n")

with open("tgt_lemma_pos_list.txt", "w") as f:
    for i in range(len(tgt_lemma_pos_list)):
        f.write(tgt_lemma_pos_list[i]+"\n")