import time
import torch
from sklearn.cluster import KMeans
import os
import pickle
import torch.nn.functional  as F
import argparse
from utils import tokenise_phrase, Encode_LM, Get_model, Identify_Indices
from tqdm import tqdm

# PADDING TEXT for XLNet to encode a short text (common practice)
PADDING_TEXT = "Bla bla bla bla bla bla bla bla " + "bla bla bla bla bla bla bla bla. <eod> </s> <eos>"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-model',
        help='model')
    parser.add_argument(
        '-mono_sent',
        required=True,
        help='mono_sent_path')
    parser.add_argument(
        '-N_sample',
        type=int,
        default=300)
    parser.add_argument(
        '-save',
        help='save_name')
    parser.add_argument(
        '-word_list',
        default=None,
        help='list of words')
    parser.add_argument(
        '-folder',
        help='save_folder')
    parser.add_argument(
        '-debug',
        action='store_true')
    parser.add_argument(
        '-K',
        type=int,
        nargs='+',
        required=True,
        help='number of clusters')
    parser.add_argument(
        '-start_from',
        type= int,
        default=-1)
    parser.add_argument(
        '-end_at',
        type= int,
        default=-1)
    parser.add_argument(
        '-skip_words',
        type=str,
        default=None)
    parser.add_argument(
        '-max_tokens',
        type=int,
        default=8192)

    opt = parser.parse_args()
    folder = None
    model_path = opt.model
    folder = opt.folder
    if not os.path.exists(folder):
        os.mkdir(folder)
    else:
        print("Directory ", folder, " already exists")

    model, tokenizer = Get_model(model_path, torch.cuda.is_available())
    stop_words_ids = []
    target_sentences = []
    embeddings = []
    input_ids = torch.cuda.LongTensor([[1, 2]])

    #Set target layers
    if model.model_name == "spanbert":
        N_layer = 25
        embdim = 1024
        target_layers = list(range(3, 23))
        tokenizer.name_or_path = "spanbert-large-cased"
    else:
        if model.model_name in ['deberta','deberta-v3',"bert","spanbert","albert","gpt2",'roberta','electra',
                                "sbert-mpnet","mpnet",'sbert-bert']:
            outputs = model(input_ids=input_ids, output_hidden_states=True)
            N_layer = len(outputs["hidden_states"])  # 13 or 25
            if N_layer == 25:
                target_layers = list(range(3, N_layer-2))  # 3~22 layers
            elif N_layer == 13:
                target_layers = list(range(3, N_layer-2))  # 3~22 layers
            else:
                raise Exception
        elif model.model_name =="fbmbart_MT":
            outputs = model.model.encoder(input_ids=input_ids,output_hidden_states=True)
            all_L_tmp = outputs["hidden_states"]  # N_layer, bs, seq_len, dim
            N_layer = len(all_L_tmp)  # 13 or 25
            assert N_layer==13
            target_layers = list(range(3, N_layer-2))  # 3~22 layers
        elif model.model_name =="xlnet":
            outputs = model(input_ids=input_ids,output_hidden_states=True)
            N_layer = len(outputs[-1])*2  # 13 or 25
            assert N_layer == 50
            target_layers = [val * 2 for val in list(range(3, 23))]  # 3~10 layers
        elif model.model_name == "fbbart":
            outputs = model(input_ids = input_ids,output_hidden_states=True)
            N_Layer_each = len(outputs[-1])
            N_layer = 3 * N_Layer_each  #enc, dec_t-1, dec_t
            if N_Layer_each ==13: #large model
                tgt_layers_base = list(range(3, N_Layer_each-2)) # 3~10
            elif N_Layer_each ==7: #base model
                tgt_layers_base = list(range(2, N_Layer_each-1)) # 2~5
            else:
                raise Exception
            target_layers = tgt_layers_base # 2~5
            target_layers += [val + 2*N_Layer_each for val in tgt_layers_base] #t
        embdim = outputs[-1][0].size()[-1]

    skip_words = []
    if opt.skip_words:
        print("Skip existing words")
        c=0
        for line in open(opt.skip_words, encoding="utf8"):
            line = line.split(" ||| ")[0].split()
            assert len(line) == embdim + 1, str(len(line))+ " " + str(embdim) + " "+ str(c) +": "+ line[0]
            skip_words.append(line[0])
            c += 1

    skip_words = set(skip_words)
    vocab = dict()
    count = -1
    layers = list(range(N_layer))
    if opt.word_list is not None:
        import os.path
        if os.path.isfile(opt.word_list):
            v = []
            for line in open(opt.word_list):
                line = line.rstrip('\n')
                v.append(line)
        else:
            v = [opt.word_list.split()]
        with open(opt.mono_sent, 'rb') as f:
            phrase2sent_1B_tmp = pickle.load(f)
        phrase2sent = dict()
        N_found= 0
        for i in tqdm(range(len(v))):
            phrase = v[i]
            if phrase not in phrase2sent_1B_tmp:
                print(phrase + "not found")
            else:
                N_found += 1
                phrase2sent[phrase] = phrase2sent_1B_tmp[phrase]
        print(N_found)
        del phrase2sent_1B_tmp
    else:
        with open(opt.mono_sent, 'rb') as f:
            phrase2sent = pickle.load(f)
        v = list(phrase2sent.keys())

    if opt.end_at == -1:
        opt.end_at = float("inf")

    start = time.time()
    N_not_found = 0
    v_remains = []
    for w in v:
        if w in skip_words:
            print("skip " + w)
            continue
        else:
            v_remains.append(w)
    print(str(len(v_remains)) + "words")

    V_size = 0
    with open(opt.folder + "/count.txt", "w") as f_count:
        with open(opt.folder + "/vec.txt", "w") as f_vec:
            with torch.no_grad():
                for i in tqdm(range(len(v_remains))):
                    phrase = v_remains[i]
                    assert phrase[0] != " "
                    count += 1
                    if opt.start_from > count:
                        continue
                    if count >= opt.end_at:
                        break
                    if count %1000 ==0:
                        print(count)
                    sentences_untok = phrase2sent[phrase]
                    assert isinstance(sentences_untok, list)
                    if len(sentences_untok)>=1:
                        sentences_untok = sentences_untok[:opt.N_sample]
                        if model.model_name == 'gpt2':
                            for j in range(len(sentences_untok)):
                                sentences_untok[j] = "<|endoftext|>" + sentences_untok[j]
                        elif model.model_name == 'xlnet':
                            for j in range(len(sentences_untok)):
                                sentences_untok[j] = PADDING_TEXT + sentences_untok[j] # a common practice when encoding a short text using XLNet
                        # elif model.model_name =="spanbert":
                        #     sentences_tmp = SpanBert_tok_CLS_SEP(tokenizer, sentences_untok)
                        sentences_tmp = tokenizer(sentences_untok)["input_ids"]
                        sentences = []
                        for x in sentences_tmp:
                            assert len(x) < 512
                            sentences.append(x)
                        phrase = " " + phrase #add space
                        phrase_tokenised_ids = tokenise_phrase(model, tokenizer, phrase)
                        target_ids = phrase_tokenised_ids
                        target_ids_replace = phrase_tokenised_ids
                        sentences_masked, mask_row_idx, mask_col_idx = \
                            Identify_Indices(sentences, target_ids, target_ids_replace, tokenizer)

                        if len(sentences_masked)>max(opt.K): #there should be more sentences than the cluster size
                            V_size+=1
                            phrase_all_states = Encode_LM(
                                tokenizer, model, sentences_masked, mask_col_idx, max_tokens= opt.max_tokens, layers = layers)
                            #phrase_all_states: N_layer, bs, n_word, dim
                            veckey = "▁".join(phrase.lstrip(" ").split(" "))
                            f_vec.write(veckey + " ")
                            k_class_list = []
                            concat_phrase_states = torch.cat([F.normalize(phrase_all_states[x].mean(dim=1),dim=-1) for x in target_layers], dim=-1)  # bs, X*dim
                            concat_phrase_states = concat_phrase_states.data.cpu().numpy()
                            for _k in opt.K: #apply K-means with K=y
                                km = KMeans(n_clusters=_k)
                                sent_class = km.fit_predict(concat_phrase_states)
                                k_class_list.append(sent_class)
                            for _k in range(len(k_class_list)): #save cluster id for each sent (for the purpose of analysis)
                                f_count.write(veckey + " " + str(len(sentences_untok)) + " " + " ".join([str(e) for e in k_class_list[_k]]) + "\n")
                            phrase_all_states = phrase_all_states.mean(dim=2)  #L, bs, dim
                            phrase_all_states = phrase_all_states[target_layers].mean(0).unsqueeze(0)
                            phrase_all_states_mean = phrase_all_states.mean(dim=1).data.cpu().numpy()  # 1, dim
                            phrase_all_states = phrase_all_states.transpose(1, 0) #bs, 1, dim
                            phrase_all_states = phrase_all_states.data.cpu().numpy()

                            phrase_states_K_list =[]
                            for _k in range(len(k_class_list)):
                                #decontextualised embs when K = _k
                                phrase_states_K_list.append([phrase_all_states[k_class_list[_k] == i].mean(0) for i in range(opt.K[_k])]) # K, L,dim

                            assert len(phrase_all_states_mean)==1
                            for l in range(len(phrase_all_states_mean)): #without clustering
                                f_vec.write(" ".join([str(x) for x in phrase_all_states_mean[l]]) + " ||| ")
                                # if opt.K > 1:
                                assert len(k_class_list) == len(opt.K)
                                for _k in range(len(k_class_list)): #with clustering
                                    phrase_states_K = phrase_states_K_list[_k]
                                    assert len(phrase_states_K[0]) == 1
                                    for i in range(opt.K[_k]):
                                        f_vec.write(" ".join([str(x) for x in phrase_states_K[i][l]]) + " ||| ")
                            f_vec.write("\n")
                elapsed_time = time.time() - start
                print("Train elapsed_time: "+str(elapsed_time))

    # save embeddings for each cluster
    K_list = [1] + opt.K
    files = []
    for K in K_list:  # [1, 2, 4, 8]:
        if not os.path.exists(opt.folder + "/K" + str(K)):
            os.mkdir(opt.folder + "/K" + str(K))
        for j in range(K):
            if not os.path.exists(opt.folder + "/K" + str(K) + "/K" + str(j)):
                os.mkdir(opt.folder + "/K" + str(K) + "/K" + str(j))
            savef = open(opt.folder + "/K" + str(K) + "/K" + str(j) + "/vec.txt", "w")
            savef.write(str(V_size) + " " + str(embdim) + "\n")
            files.append(savef)
    c = 0
    for line in open(opt.folder + "/vec.txt"):
        line = line.rstrip('\n').rstrip(" ").rstrip(" |||")
        line = line.split(" ||| ")  #
        w = None
        assert len(line) == sum(K_list), str(len(line))+ " " + str(sum(K_list)) # *N_emb
        c += 1
        count = 0
        for K in K_list:
            for j in range(K):
                line_tmp = line[count].split(' ')
                if count == 0:
                    w = line_tmp[0]
                    vec = line_tmp[1:]
                else:
                    vec = line_tmp
                files[count].write(w + " " + " ".join(vec) + "\n")
                count += 1
    for l in range(len(files)):
        files[l].close()

    #remove the original file
    os.remove(opt.folder + "/vec.txt")



