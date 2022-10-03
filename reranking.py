import sys
sys.path.insert(1, '../')
import torch
import os
import numpy as np
import pickle
import torch.nn.functional  as F
import argparse
from utils import tokenise_phrase, Encode_LM, Get_model, SpanBert_tok_CLS_SEP, Identify_Indices, Read_tgtsent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-model',
        help='model')
    parser.add_argument(
        '-tgt_sent',
        required=True,
        help='tgt_sent_path')
    parser.add_argument(
        '-candidates',
        type=str,
        required= True)
    parser.add_argument(
        '-folder',
        help='save_name')
    parser.add_argument(
        '-debug',
        action='store_true',
        help='save_name')
    parser.add_argument(
        '-out_allL',
        action='store_true',
        help='save_name')
    parser.add_argument(
        '-print_info',
        action='store_true',
        help='save_name')

    opt = parser.parse_args()
    folder = None
    save = opt.model.replace("/","_")
    model_path = opt.model
    folder = opt.folder

    if not os.path.exists(folder):
        os.mkdir(folder)
    else:
        print("Directory ", folder, " already exists")
    model, tokenizer = Get_model(model_path, torch.cuda.is_available())

    input_ids = torch.cuda.LongTensor([[1, 2]])
    if model.model_name == "spanbert":
        N_layer = 25
        if opt.out_allL:
            tgt_layers = list(range(N_layer))
        else:
            tgt_layers = list(range(3, N_layer - 2))

    elif model.model_name in ['deberta-v3',"bert", 'electra', 'sbert-bert']:
        outputs = model(input_ids=input_ids, output_hidden_states=True)
        N_layer = len(outputs["hidden_states"])  # 13 or 25
        if opt.out_allL:
            tgt_layers = list(range(N_layer))
        else:
            tgt_layers = list(range(3, N_layer - 2))

    elif model.model_name in ["fbmbart_MT", "sbert-mpnet", 'mpnet']:
        N_layer = 13
        if opt.out_allL:
            tgt_layers = list(range(N_layer))
        else:
            tgt_layers = list(range(3, N_layer - 2))
    elif model.model_name in ["marian"]:
        N_layer = 7
        if opt.out_allL:
            tgt_layers = list(range(N_layer))
        else:
            tgt_layers = list(range(2, N_layer - 1))
    elif model.model_name == "xlnet":
        N_layer = 13
        if opt.out_allL:
            tgt_layers = [val * 2 for val in list(range(25))]  # 3~10 layers
        else:
            tgt_layers = [val * 2 for val in list(range(3, 23))]  # 3~10 layers
    elif model.model_name == "fbbart":
        outputs = model(input_ids=input_ids, output_hidden_states=True)
        N_Layer_each = len(outputs[-1])
        if opt.out_allL:
            tgt_layers = list(range(3*N_Layer_each))  # 3~10
        else:
            tgt_layers_base = list(range(3, N_Layer_each - 2))  # 3~10
            tgt_layers = tgt_layers_base  # 2~5
            tgt_layers += [val + 2 * N_Layer_each for val in tgt_layers_base]  # t
    else:
        raise Exception

    target_sentences = Read_tgtsent(opt.tgt_sent, model)

    with open(opt.candidates, 'rb') as f:
        para_candidates = pickle.load(f) # load substitute cands
        assert len(target_sentences) == len(para_candidates)

    candidates2outer_score_list = []
    if opt.out_allL:
        candidates2outer_score_all_list = []
    with torch.no_grad():
        for sid, line in enumerate(target_sentences):
            para_candidates_tmp = para_candidates[sid]
            candidates_word2score = dict()
            if opt.out_allL:
                candidates_word2score_all= dict()
            #line: phrase, mased sent
            phrase = line[0]
            sentences_untok = line[1:]
            assert len(sentences_untok) == 1
            if model.model_name == "deberta-v3":
                if " [MASK]" in sentences_untok[0]:
                    token = " [MASK]"
                else:
                    token = "[MASK]"
                sentences_untok = [x.replace("<mask>", "[MASK]") for x in sentences_untok]
            else:
                if " <mask>" in sentences_untok[0]:
                    token = " <mask>"
                else:
                    token = "<mask>"
            if model.model_name =='spanbert':
                sentences = SpanBert_tok_CLS_SEP(tokenizer, sentences_untok)
            else:
                sentences = tokenizer(sentences_untok)["input_ids"] #tokenise

            phrase = " " + phrase
            if model.model_name=='spanbert':
                mask_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token.lstrip(" ")))
            else:
                mask_ids = tokenizer(token, add_special_tokens=False)["input_ids"]  # remove CLS/SEP
            phrase_tokenised_ids = tokenise_phrase(model, tokenizer, phrase)
            if opt.print_info:
                print("******************************")  #
                print(" ".join(tokenizer.convert_ids_to_tokens(phrase_tokenised_ids)))
            len2words = {}
            for w in list(para_candidates_tmp.keys()): #each para candidate
                w_for_tok = " " + w
                cand_ids = tokenise_phrase(model, tokenizer, w_for_tok)
                if len(cand_ids) not in len2words:
                    len2words[len(cand_ids)] = [(w, cand_ids)]
                else:
                    len2words[len(cand_ids)].append((w, cand_ids))

            # replace masked id in sentences with the target word id (phrase_tokenised_ids)
            # and return index
            sentences_masked, mask_row_idx, mask_col_idx = \
                Identify_Indices(sentences, mask_ids, phrase_tokenised_ids, tokenizer)

            phrase_all_states = \
                Encode_LM(tokenizer, model, sentences_masked, mask_col_idx,
                                max_tokens=8192, layers=tgt_layers)

            # phrase_all_states: f^l(x,c); L, 1, n_tokens, dim
            phrase_all_states = phrase_all_states.mean(1).mean(1)  # L, dim
            phrase_all_states_norm = F.normalize(phrase_all_states, dim=-1)  # L, dim
            word_list = []
            for wlen in len2words.keys():
                mask_ids_replace = [tokenizer.mask_token_id for _ in range(wlen)] # Placeholder
                #replace <mask> with wlen*<mask>
                sentences_masked, mask_row_idx, mask_col_idx= \
                    Identify_Indices(sentences, mask_ids, mask_ids_replace,tokenizer)
                candidate_ids = len2words[wlen] #candidate word/phrase with wlen tokens
                batch = []
                bs = 100
                for i in range(0, len(candidate_ids),bs):
                    batch.append(candidate_ids[i:i+bs])
                for candidate_ids in batch:
                    new_input_ids = []
                    assert len(sentences_masked)==1
                    cossim_score = None
                    for word_and_id in candidate_ids:
                        word = word_and_id[0]
                        phraseids = np.array(word_and_id[1]).reshape(1,-1) #1,w_len
                        input_ids_tmp = np.array(sentences_masked.copy())
                        # Replace placeholder (<mask>s) with candidate words
                        input_ids_tmp[mask_row_idx, mask_col_idx] = phraseids
                        new_input_ids.append(input_ids_tmp.tolist()[0])

                    mask_col_idx_tmp = np.array([mask_col_idx[0] for _ in range(len(new_input_ids))])  # bs, N_tok
                    #new_input_ids: bs, seq_len
                    phrase_all_states_tmp = \
                        Encode_LM(tokenizer, model, new_input_ids, mask_col_idx_tmp,
                                        max_tokens=8192, layers=tgt_layers)
                    # phrase_all_states_tmp: f^l(y,c), L, bs, n_tok, dim
                    phrase_all_states_tmp = phrase_all_states_tmp.mean(2) #L, bs, dim (mean over tokens)
                    L, bs, edim = phrase_all_states_tmp.size()
                    phrase_all_states_tmp = F.normalize(phrase_all_states_tmp, dim=-1)
                    cossim_score = phrase_all_states_tmp * \
                                   phrase_all_states_norm.unsqueeze(1).repeat(1, len(new_input_ids), 1) #L, bs, dim
                    if opt.out_allL:
                        cossim_score_all = cossim_score.sum(dim=-1).data.cpu().numpy().round(5)  # L, bs
                        for i, word_and_id in enumerate(candidate_ids):
                            word = word_and_id[0]
                            assert word not in candidates_word2score_all
                            # if len(cossim_score_all.shape)==3:
                            #     candidates_word2score_all[word] = cossim_score_all[:,i] #L, dim
                            # else:
                            assert len(cossim_score_all.shape) == 2
                            candidates_word2score_all[word] = cossim_score_all[:,i] #L, dim
                    # cossim_score: L, bs, dim
                    cossim_score = cossim_score.sum(dim=-1).mean(dim=0)  # bs: (mean over tgt layers)
                    assert len(cossim_score) == len(candidate_ids)
                    for i, word_and_id in enumerate(candidate_ids):
                        word = word_and_id[0]
                        word_list.append(word)
                        assert word not in candidates_word2score
                        candidates_word2score[word] = round(cossim_score[i].data.cpu().tolist(),3)

            if opt.out_allL:
                candidates2outer_score_all_list.append(candidates_word2score_all)

            new_candidates_word2score = {k: v for k, v in sorted(candidates_word2score.items(), key=lambda item: -1*item[1])}
            candidates2outer_score_list.append(new_candidates_word2score)

    if not opt.debug:
        with open(opt.folder + "/" + save + "_candidates2reranking_score.pkl", 'wb') as f:
            pickle.dump(candidates2outer_score_list, f)
        if opt.out_allL:
            with open(opt.folder + "/" + save + "_candidates2reranking_score_all.pkl", 'wb') as f:
                pickle.dump(candidates2outer_score_all_list, f)


