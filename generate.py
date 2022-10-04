import torch
import os
import numpy as np
import Levenshtein
import pickle
import torch.nn.functional  as F
import argparse
from utils import Encode_LM, Get_model, Identify_Indices, Read_tgtsent, Read_Embfiles
import sys
sys.path.insert(1, '../')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-model',
        help='model')
    parser.add_argument(
        '-tgt_sent',
        required=True)
    parser.add_argument(
        '-beam_size',
        type=int,
        default=50)
    parser.add_argument(
        '-tgt_layer',
        type=int,
        default=None)
    parser.add_argument(
        '-folder')
    parser.add_argument(
        '-debug',
        action='store_true')
    parser.add_argument(
        '-lambda_val',
        type=float,
        default=0.7)
    parser.add_argument(
        '-print_info',
        action='store_true')
    parser.add_argument(
        '-lev',
        type=float,
        default=0.5)
    parser.add_argument(
        '-vec',
        type=str,
        default=[],
        nargs='+')

    opt = parser.parse_args()
    folder = None
    model_path = opt.model
    folder = opt.folder
    if not os.path.exists(folder):
        os.mkdir(folder)
    else:
        print("Directory ", folder, " already exists")
    model, tokenizer = Get_model(model_path, torch.cuda.is_available())

    input_ids = torch.cuda.LongTensor([[1, 2]])
    if model.model_name=="spanbert":
        raise NotImplementedError
        # N_layer = 25
        # tgt_layers = list(range(3, N_layer - 2))
    elif model.model_name in ['jbert',"bert",'electra','sbert-bert','deberta-v3']:
        outputs = model(input_ids=input_ids, output_hidden_states=True)
        N_layer = len(outputs["hidden_states"])  # 13 or 25
        tgt_layers = list(range(3, N_layer-2))
    elif model.model_name in ["fbmbart_MT","sbert-mpnet",'mpnet']:
        tgt_layers = list(range(3,11))
    elif model.model_name =="xlnet":
        tgt_layers = [val * 2 for val in list(range(3, 23))]  # 3~10 layers
    elif model.model_name == "fbbart":
        outputs = model(input_ids=input_ids, output_hidden_states=True)
        N_Layer_each = len(outputs[-1]) #13
        tgt_layers_base = list(range(3, N_Layer_each - 2))  # 3~10
        tgt_layers = tgt_layers_base # 2~5
        tgt_layers += [val + 2 * N_Layer_each for val in tgt_layers_base] #t
    else:
        raise Exception
    if opt.tgt_layer:
        if model.model_name =="xlnet":
            tgt_layers = [2*opt.tgt_layer]
        else:
            tgt_layers = [opt.tgt_layer]

    if model.model_name == 'fbbart':# Average encoder and decoder scores
        N_vec = len(opt.vec)
        assert N_vec % 2 == 0
        Vocab, word2id, emb_list1 = Read_Embfiles(opt.vec[:N_vec // 2])
        Vocab2, _, emb_list2 = Read_Embfiles(opt.vec[N_vec // 2:])
        assert all(Vocab2 == Vocab)
        emb_list = [F.normalize(emb_list1, dim=-1), F.normalize(emb_list2, dim=-1)]
    else:
        Vocab, word2id, emb_list = Read_Embfiles(opt.vec)
        emb_list = [F.normalize(emb_list, dim=-1)]

    if model.model_name == "spanbert":
        model.device = emb_list[0].device

    stop_words_ids = []
    target_sentences = Read_tgtsent(opt.tgt_sent, model)
    candidates2cossim_score_list = []
    candidates2decntxt_score_list = []
    candidates_word2score_vec_list = []
    save_file = opt.model.replace("/", "_") + "_beam_" + str(opt.beam_size) + "lambda_val" + str(opt.lambda_val)
    with torch.no_grad():
        for s_id, line in enumerate(target_sentences):
            # line: word, masked sentence
            phrase = line[0]
            input_sentence = line[1:]
            assert len(input_sentence)==1
            first_sent = input_sentence[0]
            if model.model_name=="deberta-v3":
                first_sent.replace("<mask>","[MASK]")
                input_sentence = [x.replace("<mask>", "[MASK]") for x in input_sentence]
                if " [MASK]" in first_sent:
                    token = " [MASK]"
                else:
                    token = "[MASK]"
            else:
                if " <mask>" in first_sent:
                    token = " <mask>"
                else:
                    token = "<mask>"
            if model.model_name =="spanbert":
                mask_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
            elif model.model_name in ['gpt2', 'xlnet']:
                mask_ids = tokenizer(token, add_special_tokens=False)["input_ids"]  # remove CLS/SEP
            else:
                mask_ids = tokenizer(token)["input_ids"][1:-1]  # remove CLS/SEP

            if model.model_name == 'spanbert':
                raise NotImplementedError
                # sentences = SpanBert_tok_CLS_SEP(tokenizer, input_sentence)
            else:
                sentences = tokenizer(input_sentence)["input_ids"]

            if len(sentences[0])>512:
                print("target sentence length is greater than 512")
                raise Exception
            phrase_tokenised_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(phrase)) #remove CLS/SEP
            phrase_tokenised_str = tokenizer.convert_ids_to_tokens(phrase_tokenised_ids)
            if opt.print_info:
                print(phrase_tokenised_ids)
                print("******************************")  #
                print(" ".join(phrase_tokenised_str))  #

            sentences_masked, mask_row_idx, mask_col_idx = \
                Identify_Indices(sentences, mask_ids, phrase_tokenised_ids, tokenizer)
            gold_sent = " ".join(tokenizer.convert_ids_to_tokens(sentences_masked[0]))
            if opt.print_info:
                print(gold_sent)
            phrase_all_states= \
                Encode_LM(tokenizer, model, sentences_masked, mask_col_idx,
                            max_tokens=8192, layers=tgt_layers)
            #phrase_all_states: L, bs, n_tok, dim
            candidates_word2score = dict()
            candidates_word2score_vec = dict()
            if model.model_name == 'fbbart' and len(tgt_layers) == 16: #average enc-dec scores
                assert len(emb_list) == 2
                N_half = len(phrase_all_states) // 2
                phrase_all_states_new = phrase_all_states[0]
                phrase_all_states_new2 = phrase_all_states[N_half]
                for _k in range(1, N_half): #enc
                    phrase_all_states_new += phrase_all_states[_k]
                for _k in range(N_half + 1, len(phrase_all_states)): #dec
                    phrase_all_states_new2 += phrase_all_states[_k]
                del phrase_all_states
                phrase_all_states = [phrase_all_states_new, phrase_all_states_new2]  #2, 1, n_tok, dim
            else:
                assert len(emb_list) == 1
                phrase_all_states_new = phrase_all_states[0]
                for _k in range(1, len(phrase_all_states)):
                    phrase_all_states_new += phrase_all_states[_k]
                del phrase_all_states
                phrase_all_states = phrase_all_states_new.unsqueeze(0) #1, 1, n_tok, dim
            orig_masking = torch.zeros(len(word2id)).to('cuda')
            if phrase.lstrip(" ") in word2id:#assing -inf score to the tgt word
                phrase_vec_id = word2id[phrase.lstrip(" ")]  #decontext emb id
                orig_masking[phrase_vec_id] = float('-inf')
            elif phrase.lower().lstrip(" ") in word2id:
                phrase_vec_id = word2id[phrase.lower().lstrip(" ")]
                orig_masking[phrase_vec_id] = float('-inf')
            else:
                phrase_vec_id = None
                print(phrase + " Not Found in decontxt emb")

            phrase_len = len(phrase.lstrip(" ").rstrip(" "))
            if opt.lev > 0: #edit-distance heuristic
                for _k, w in enumerate(Vocab):
                    if Levenshtein.distance(w.lower(), phrase.lower().lstrip(" ").rstrip(" "))/max(len(w), len(phrase.lstrip(" ").rstrip(" "))) < opt.lev:
                        orig_masking[_k] = float('-inf') #discard similar-spelling words

            add_count = 0
            cossim_x_yk_all = None
            cossim_x_xk = torch.zeros(len(emb_list[0])).to(phrase_all_states[0].device) #k
            if model.model_name == "fbbart" and len(tgt_layers)== 16:
                assert len(emb_list)== 2
            else:
                assert len(emb_list) == 1
            assert len(phrase_all_states)==len(emb_list)
            for k in range(len(emb_list)):
                emb_normalised = emb_list[k] #deconx_emb: K, V
                add_count += 1
                #phrase_all_states: len(emb_list), 1, n_tok, dim
                tgt_phrase_emb = phrase_all_states[k].mean(dim=0).mean(dim=0)  # dim
                tgt_phrase_emb = F.normalize(tgt_phrase_emb, dim=-1) # dim
                k_class, V_size, _ = emb_normalised.size()
                cossim_x_yk = [emb_normalised[k] * tgt_phrase_emb.view(1, -1).repeat(V_size, 1) for k in range(k_class)]  # k, V, dim
                cossim_x_yk = torch.stack(cossim_x_yk)# k, V, dim
                cossim_x_yk = torch.sum(cossim_x_yk,dim=-1) #k, V
                if phrase_vec_id is not None: #if phrase in decontx emb
                    phrase_devec = F.normalize(emb_normalised[:, phrase_vec_id], dim=-1) # k, dim
                    cossim_x_xk += torch.sum(phrase_devec * tgt_phrase_emb.view(1, -1).repeat(k_class,1), dim=-1)  # k
                if cossim_x_yk_all is None:
                    cossim_x_yk_all = cossim_x_yk  # k, V
                else:
                    cossim_x_yk_all += cossim_x_yk # k, V
            #calc global similarity
            global_sim = torch.zeros(cossim_x_yk_all.shape).to(model.device) #k, V
            if phrase_vec_id is not None: #if phrase in decontx emb
                _, cluster_idx_phrase = cossim_x_xk.max(dim=0) # closest centroid to f(x,c)
                for k in range(len(emb_list)):
                    emb_normalised = emb_list[k]#.to(model.device) #k, V, dim
                    phrase_devec = emb_normalised[:, phrase_vec_id]  # k, dim
                    phrase_devec_centroid = phrase_devec[cluster_idx_phrase]  #fjc(x): dim
                    phrase_devec_centroid = phrase_devec_centroid.view(1, -1).expand_as(emb_normalised[0])  # V, dim
                    cossim = [emb_normalised[k] * phrase_devec_centroid for k in range(k_class)]  # k, V
                    cossim = torch.stack(cossim) #k, V, dim
                    global_sim += torch.sum(cossim, dim=-1) #k, V
                    cossim_x_yk_all = opt.lambda_val * cossim_x_yk_all + (1 - opt.lambda_val) * global_sim

            cossim_x_yk_all, cluster_idx = cossim_x_yk_all.max(dim=0)  # V (maxmimum scores)
            cossim_x_yk_all = cossim_x_yk_all/add_count
            cossim_x_yk_all = cossim_x_yk_all + orig_masking # mask certain words
            top_idx = cossim_x_yk_all.topk(opt.beam_size)  # 50
            pred_words = np.array(Vocab[top_idx[1].data.cpu().tolist()])
            values = top_idx[0].data.cpu().tolist()
            idx = top_idx[1]
            if opt.print_info:
                print("pred words ALL")
                print(pred_words[0:5])
                print(values[0:5])
            for j, word in enumerate(pred_words):
                candidates_word2score[word] = values[j]
            candidates2cossim_score_list.append(candidates_word2score)

    if not opt.debug:
        with open(opt.folder + "/" + save_file + "_candidates2cossim_score.pkl", 'wb') as f:
            pickle.dump(candidates2cossim_score_list, f)
