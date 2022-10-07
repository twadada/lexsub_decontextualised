import pickle
import timeit
import argparse

def Extract_Sentences(phrasekey_list, file, save, N_sent, min_Nword):
    phrase_len = 1
    phrasekey_list_tmp = phrasekey_list.copy()
    phrase2sent ={}
    for sentence in open(file):
        sentence = sentence.strip('\n')
        word_list = sentence.split(" ")
        if len(word_list) < 200 and len(word_list)>min_Nword: #discard too long/short sentences (for lowering computational cost/reducing noise)
            for i in range(1, len(word_list) - phrase_len+1):
                if tuple(word_list[i:i + phrase_len]) in phrasekey_list_tmp:
                    phrase = " ".join(word_list[i:i + phrase_len])
                    if phrase not in phrase2sent:
                        phrase2sent[phrase] = set([sentence])
                    elif len(phrase2sent[phrase])< N_sent:
                        phrase2sent[phrase].add(sentence)
                    else:
                        if phrase in phrasekey_list_tmp:
                            phrasekey_list_tmp.remove(phrase)

        if len(phrasekey_list_tmp) ==0:
            break

    for x in phrase2sent.keys():
        phrase2sent[x] = list(phrase2sent[x])
    with open(save + ".pkl", 'wb') as f:
        pickle.dump(phrase2sent, f, pickle.HIGHEST_PROTOCOL)
    return phrase2sent

parser = argparse.ArgumentParser()

parser.add_argument(
    '-wordfile',
    required=True,
    help='silver_sent_path')

parser.add_argument(
    '-monofile',
    required=True,
    help='silver_sent_path')

parser.add_argument(
    '-min_Nword',
    type=int,
    default=15)

parser.add_argument(
    '-N_sent',
    type=int,
    default=2000)

parser.add_argument(
    '-folder',
    required=True,
    help='silver_sent_path')

opt = parser.parse_args()
wordfile = opt.wordfile
folder = opt.folder

phrase2sent  = {}
phrasekey_list = set([])
for line in open(wordfile): #list of words
    line = line.strip('\n') #word
    phrase = line.split()
    assert len(phrase) == 1
    phrasekey_list.add(tuple(phrase))

print("#phrase, ",len(phrasekey_list))

N_count = 0
save = folder + "/"  + wordfile.split("/")[-1]

start = timeit.default_timer()
phrases2sent = Extract_Sentences(phrasekey_list, opt.monofile, save+"_silversent", opt.N_sent, opt.min_Nword)
stop = timeit.default_timer()
print('Time: ', stop - start)
print("#phrase, ", len(phrases2sent))