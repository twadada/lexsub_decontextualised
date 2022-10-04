from transformers.models.roberta.modeling_roberta import gelu as roberta_gelu
from transformers.models.mpnet.modeling_mpnet import gelu as mpnet_gelu
from transformers.models.electra.modeling_electra import get_activation as electra_get_activation
from transformers.models.bart.modeling_bart import shift_tokens_right as mono_shift_tokens_right
# import sys
# sys.path.insert(1, '../')
import torch
import numpy as np
from transformers import XLNetLMHeadModel, XLNetTokenizer,ElectraTokenizer,ElectraModel,ElectraForPreTraining,MBart50TokenizerFast,MBartForConditionalGeneration,T5Tokenizer, BartTokenizer,AlbertTokenizer, GPT2Tokenizer, OpenAIGPTTokenizer,OpenAIGPTLMHeadModel, GPT2LMHeadModel, AlbertForMaskedLM, BartForConditionalGeneration, XLMWithLMHeadModel, CamembertForMaskedLM, BertForMaskedLM, BertTokenizer, BertModel, AutoTokenizer, AutoModel, RobertaForMaskedLM, T5ForConditionalGeneration, T5EncoderModel, M2M100ForConditionalGeneration,M2M100Tokenizer,MPNetTokenizer,MPNetForMaskedLM, PegasusTokenizer,PegasusForConditionalGeneration
# from SpanBERT.code.pytorch_pretrained_bert.tokenization import BertTokenizer as SpanBertTokenizer
# from SpanBERT.code.pytorch_pretrained_bert.modeling import BertForMaskedLM_SBO as SpanBertForMaskedLM_SBO

#PADDING TEXT for XLNet to encode a short text (common practice)
PADDING_TEXT = "Bla bla bla bla bla bla bla bla bla bla bla bla bla bla bla bla. <eod> </s> <eos>"


#
# def SpanBert_tok_CLS_SEP(tokenizer, sent_list):
#     out = []
#     for sent in sent_list:
#         sent_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent))
#         if len(sent_id)<510:
#             sent_id = [tokenizer.cls_token_id]+ sent_id +[tokenizer.sep_token_id]
#             out.append(sent_id)
#         else:
#             print(sent)
#             print(tokenizer.tokenize(sent))
#     return out
#


def load_w2v(file, word_list = None):
    word2vec = {}
    if word_list is not None:
        word_list = set(word_list)
    with open(file, 'r', errors='ignore') as f:
        first_line = f.readline()
        first_line=first_line.split(' ')
        assert len(first_line)==2
        dim = int(first_line[1])
        for line in f:
            line = line.rstrip('\n')
            line = line.rstrip(' ')
            w = line.split(' ')[0]
            vec = line.split(' ')[1:]
            if word_list:
                if w in word_list:
                    word2vec[w] = [float(x) for x in vec]
            else:
                word2vec[w] = [float(x) for x in vec]
            # if N_words and len(word2vec) == N_words:
            #     break
    return word2vec

def Read_tgtsent(tgt_sent, model):
    target_sentences=[]
    for line in open(tgt_sent, encoding="utf8"):
        line = line.rstrip("\n").split("\t") #phrase, masked sent
        assert len(line)==2
        if model.model_name == 'gpt2':
            for i in range(1,len(line)):
                line[i] = "<|endoftext|>"+line[i]
        elif model.model_name == 'xlnet':
            for i in range(1,len(line)):
                line[i] = PADDING_TEXT + line[i]
        target_sentences.append(line)
    return target_sentences


def Read_Embfiles(vec_folders):
    # vec_folders: K
    emb_list = [] #L
    Vocab = None
    for i in range(len(vec_folders)):
        emb = []
        Vocab_tmp = []
        file = vec_folders[i] #bert-large-uncased_k4/K4/K0
        vec_tmp = load_w2v(file, word_list=None)
        for w in vec_tmp.keys():
            Vocab_tmp.append(w)
            emb.append(vec_tmp[w])
        emb = torch.FloatTensor(np.array(emb)) #V, dim
        # emb_list_orig[j].append(emb)
        emb_list.append(emb) #k, V, dim
        if Vocab is None:
            Vocab = np.array(Vocab_tmp)
        else:
            assert all(Vocab == np.array(Vocab_tmp))
    emb_list = torch.stack(emb_list, dim=0).to("cuda")  # k, V, emb
    word2id = {}
    for w in Vocab:
        word2id[w] = len(word2id)
    print("V: ",len(Vocab))
    return Vocab, word2id, emb_list


def Identify_Indices(tokenised_sentence_ids, phrase_ids, mask_ids, tokenizer):
    phrase_len = len(phrase_ids)
    assert phrase_len != 0
    phrase_tmp = phrase_ids.copy()
    phrase_col_idx = []
    out = []
    is_xlnet = tokenizer.name_or_path == "xlnet-large-cased"
    phrase_additional = []
    if is_xlnet:
        #XLNet tokeniser can produce different tokens depending on the surrounding context.
        if tokenizer.unk_token_id not in phrase_ids: #NO <unk>
            phrase_str = tokenizer.convert_ids_to_tokens(phrase_ids)
            phrase_str = "".join(phrase_str).replace("▁","")
            phrase_tmp = tokenizer(phrase_str, add_special_tokens=False)["input_ids"]
            assert phrase_tmp == phrase_ids, " ".join(tokenizer.convert_ids_to_tokens(phrase_tmp)) + " " + phrase_str
            phrase2 = tokenizer("a "+phrase_str,add_special_tokens=False)["input_ids"][1:]
            phrase3 = tokenizer("aa "+phrase_str,add_special_tokens=False)["input_ids"][2:]
            phrase4 = tokenizer(phrase_str +" a",add_special_tokens=False)["input_ids"][:-1]
            phrase5 = tokenizer(phrase_str +" aa",add_special_tokens=False)["input_ids"][:-2]
            phrase_set = set([tuple(phrase_tmp), tuple(phrase2),tuple(phrase3),tuple(phrase4),tuple(phrase5)])
            if len(phrase_set) == 1: #all segmentations are the same
                pass
            else:
                phrase_additional_all = phrase_set - set([tuple(phrase_tmp)])
                phrase_additional_all = list(phrase_additional_all) #[(tuple),(tuple)]
                for x in phrase_additional_all: #for each tuple
                    phrase_additional.append(list(x))
    for sid, sent in enumerate(tokenised_sentence_ids):
        flag = False
        sent_tmp = sent.copy()
        other_tok_found = False
        for i in range(len(sent_tmp)-len(phrase_tmp)+1):
            if len(phrase_additional):
                for e in range(len(phrase_additional)):
                    if all([sent_tmp[i + j] == phrase_additional[e][j] for j in range(len(phrase_additional[e]))]):
                        other_tok_found = True
                        to_fill = phrase_additional[e]
                        break
            if all([sent_tmp[i + j] == phrase_tmp[j] for j in range(len(phrase_tmp))])\
            or other_tok_found:
                flag = True
                phrase_idx = [i + j for j in range(len(mask_ids))]
                phrase_col_idx.append(phrase_idx)
                for j in range(phrase_len):  # remove
                    sent_tmp.pop(i)
                if other_tok_found:
                    assert mask_ids == phrase_ids
                    for j in range(len(to_fill)):  # remove
                        sent_tmp.insert(i + j, to_fill[j])
                else:
                    for j in range(len(mask_ids)):  # remove
                        sent_tmp.insert(i+j, mask_ids[j])
                break
        if not flag:
            print(tokenizer.convert_ids_to_tokens(sent))
            print(tokenizer.convert_ids_to_tokens(phrase_ids))
            print("NOT FOUND")
            raise Exception
        out.append(sent_tmp)
    phrase_col_idx = np.array(phrase_col_idx)  # bs, N_mask
    phrase_row_idx = np.arange(len(phrase_col_idx))[:, None]  # bs, 1
    return out, phrase_row_idx, phrase_col_idx

def tokenise_phrase(model, tokenizer, phrase):
    assert phrase[0] == " " #add space for SentencePiece
    if model.model_name == 'spanbert':
        premask_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(phrase.lstrip(" ")))
    else:
        premask_ids = tokenizer(phrase, add_special_tokens=False)["input_ids"]  # remove CLS/SEP
        if model.model_name in ['deberta','deberta-v3',"bert","electra", 'fbmbart','fbmbart_MT', 'albert', 'mpnet', 'sbert-bert', 'sbert-mpnet','fbbart', 'roberta', 'sbert-roberta']:
            premask_ids_tmp = tokenizer(phrase)["input_ids"][1:-1]  # remove CLS/SEP
            assert premask_ids == premask_ids_tmp
    return premask_ids

def _create_perm_mask_and_target_map(seq_len, target_ids, device, mask_tgt = False):
    """
    Generates permutation mask and target mapping.
    If `self.masked` is true then there is no word that sees target word through attention.
    If it is false then only target word doesn't see itself.
    Args:
        seq_len: length of the sequence (context)
        target_ids: target word indexes
    Returns:
        two `torch.Tensor`s: permutation mask and target mapping
    """
    # assert isinstance(target_ids[0], int), "One target per sentence"
    # assert isinstance(target_ids[0], int), "One target per sentence"
    batch_size = len(target_ids)
    pred_len = len(target_ids[0])
    perm_mask = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.float)
    # target_mapping = torch.zeros((batch_size, 1, seq_len))
    target_mapping = torch.zeros((batch_size, pred_len, seq_len))
    for idx in range(batch_size):
        target_id = target_ids[idx]
        for k, each in enumerate(target_id):
            target_mapping[idx, k, each] = 1.0
        for k, each in enumerate(target_id):
            if mask_tgt:
                perm_mask[idx, :, each] = 1.0
            else:
                perm_mask[idx, each, each] = 1.0

    perm_mask = perm_mask.to(device)
    target_mapping = target_mapping.to(device)
    return perm_mask, target_mapping

def _Encode_model(model, input_ids,token_ids,attn, col_idx):

    if model.model_name in ["spanbert", "bert","jbert"]:
        outputs = model.bert(input_ids = input_ids,
                             attention_mask= attn,
                             token_type_ids= token_ids,
                             output_hidden_states=True)
        last_layer_tmp = outputs[0].detach()  # bs, seq_len, dim
        all_L_tmp = outputs[-1]
        ffn_tmp = model.cls.predictions.transform(last_layer_tmp)#[row_idx, col_idx]

    elif model.model_name == 'xlnet':
        seq_len = input_ids.size()[1]
        perm_mask, target_mapping = _create_perm_mask_and_target_map(seq_len, col_idx, input_ids.device)
        outputs = model.transformer(
            input_ids = input_ids,
            perm_mask= perm_mask,
            target_mapping = target_mapping,
            attention_mask = attn,
            output_hidden_states = True
        )
        last_layer_tmp = outputs["last_hidden_state"] # g embeddings
        # print(outputs[-1])
        # last_layer_tmp = last_layer_tmp.repeat(1,N_phrase,1)
        all_L_tmp = outputs["hidden_states"]
        ffn_tmp = last_layer_tmp
        # predictions = predictions[:, 0, :]
    elif model.model_name == 'electra':
        # emb_tmp = input_embbeddings[i:i + bs]
        outputs = model.electra(input_ids=input_ids,
                                attention_mask=attn,
                                token_type_ids=token_ids,
                                output_hidden_states=True)
        last_layer_tmp = outputs[0].detach()  # bs, seq_len, dim
        all_L_tmp = outputs[-1]
        ffn_tmp = model.discriminator_predictions.dense(last_layer_tmp)
        ffn_tmp = electra_get_activation(model.discriminator_predictions.config.hidden_act)(ffn_tmp)
        # logits = model.discriminator_predictions.dense_prediction(ffn_tmp).squeeze(-1)

    elif model.model_name in ['deberta-v3','deberta']:
        outputs = model(input_ids=input_ids,
                        attention_mask=attn,
                        token_type_ids=token_ids,
                        output_hidden_states=True)
        last_layer_tmp = outputs["last_hidden_state"].detach()  # bs, seq_len, dim
        all_L_tmp = outputs["hidden_states"]
        ffn_tmp = last_layer_tmp
        # logits = model.discriminator_predictions.dense_prediction(ffn_tmp).squeeze(-1)

    elif model.model_name in ["roberta"]:
        # emb_tmp = input_embbeddings[i:i + bs]
        outputs = model.roberta(input_ids=input_ids,
                                attention_mask=attn,
                                output_hidden_states=True)
        last_layer_tmp = outputs[0].detach()  # bs, seq_len, dim
        all_L_tmp = outputs[-1]
        ffn_tmp = model.lm_head.dense(last_layer_tmp)  # bs, s_len, dim
        ffn_tmp = roberta_gelu(ffn_tmp)  # bs, s_len, dim
        ffn_tmp = model.lm_head.layer_norm(ffn_tmp)

    elif model.model_name in ["mpnet"]:
        # emb_tmp = input_embbeddings[i:i + bs]
        # emb_tmp = input_embbeddings[i:i + bs]
        outputs = model.mpnet(input_ids=input_ids,
                              attention_mask=attn,
                              output_hidden_states=True)
        last_layer_tmp = outputs[0].detach()  # bs, seq_len, dim
        all_L_tmp = outputs[-1]
        ffn_tmp = model.lm_head.dense(last_layer_tmp)
        ffn_tmp = mpnet_gelu(ffn_tmp)
        ffn_tmp = model.lm_head.layer_norm(ffn_tmp)#[row_idx, col_idx]  # bs, s_len, dim

    elif model.model_name.startswith("sbert"):
        # No ML head
        # emb_tmp = input_embbeddings[i:i + bs]
        outputs = model(input_ids=input_ids,
                        attention_mask=attn,
                        output_hidden_states=True)
        last_layer_tmp = outputs[0].detach()  # bs, seq_len, dim
        all_L_tmp = outputs[-1]
        ffn_tmp = last_layer_tmp#[row_idx, col_idx]

    elif model.model_name in ["albert"]:
        # emb_tmp = input_embbeddings[i:i + bs]
        outputs = model.albert(input_ids=input_ids,
                               attention_mask=attn,
                               token_type_ids=token_ids,
                               output_hidden_states=True)
        last_layer_tmp = outputs[0]  # bs, seq_len, dim
        ffn_tmp = model.predictions.dense(last_layer_tmp)
        ffn_tmp = model.predictions.activation(ffn_tmp)
        ffn_tmp = model.predictions.LayerNorm(ffn_tmp)
        ffn_tmp = ffn_tmp #[row_idx, col_idx]  # bs, N_mask, dim
        all_L_tmp = outputs[-1]
    elif model.model_name in ["marian"]:
        outputs = model.model.encoder(input_ids=input_ids,
                                      attention_mask=attn,
                                      output_hidden_states=True)
        last_layer_tmp = outputs[0]  # bs, seq_len, dim
        ffn_tmp = last_layer_tmp #[row_idx, col_idx]  # bs, N_mask, dim
        all_L_tmp = outputs['hidden_states'] #7
    elif model.model_name in ["fbmbart_MT"]:
        #
        # outputs = model.model(input_ids = input_ids,
        #                       attention_mask = attn,
        #                       output_hidden_states=True)
        outputs = model.model.encoder(input_ids = input_ids,
                              attention_mask = attn,
                              output_hidden_states=True)
        # print("enc")
        last_layer_tmp = outputs["last_hidden_state"] #bs, seq_len, dim
        ffn_tmp = last_layer_tmp  # [row_idx, col_idx]
        all_L_tmp = outputs["hidden_states"]  # N_layer, bs, seq_len, dim

        # input: lang_id, w1,...wn, EOS
        # dec_input_tmp = input_ids.clone()
        # dec_input_tmp[:, 0] = tokenizer.tgt_lang_id
        # # dec_input_tmp[row_idx, col_idx] = unk_token_id
        # dec_input_tmp = mbart_shift_tokens_right(dec_input_tmp, model.model.config.pad_token_id)
        # # dec_input: EOS, LANG_id, w1,...,wn
        # # dec_output: LANG_id, w1,..., EOS
        # outputs = model.model(input_ids=input_ids,
        #                       decoder_input_ids=dec_input_tmp,
        #                       attention_mask=attn,
        #                       output_hidden_states=True)
        # last_layer_tmp = outputs[0]
        # ffn_tmp = last_layer_tmp #[row_idx, col_idx]
        # all_L_tmp = outputs["encoder_hidden_states"]  # N_layer, bs, seq_len, dim
        # all_L_tmp = all_L_tmp + outputs["decoder_hidden_states"]  # decoder =h_t-1
        # all_L_tmp = all_L_tmp + tuple([x[:, 1:] for x in outputs["decoder_hidden_states"]])  # h_t
    elif model.model_name in ["pegasus"]:
        outputs = model.model.encoder(input_ids = input_ids,
                              attention_mask = attn,
                              output_hidden_states=True)
        # print("enc")
        last_layer_tmp = outputs["last_hidden_state"] #bs, seq_len, dim
        ffn_tmp = last_layer_tmp  # [row_idx, col_idx]
        all_L_tmp = outputs["hidden_states"]  # N_layer, bs, seq_len, dim

    elif model.model_name in ["fbbart"]:
        dec_input_tmp = input_ids.clone()
        # dec_input_tmp[row_idx, col_idx] = unk_token_id
        dec_input_tmp = mono_shift_tokens_right(dec_input_tmp, model.model.config.pad_token_id,
                                                model.model.config.decoder_start_token_id)
        outputs = model.model(input_ids=input_ids,
                              decoder_input_ids=dec_input_tmp,
                              attention_mask=attn,
                              output_hidden_states=True)
        last_layer_tmp = outputs[0]
        ffn_tmp = last_layer_tmp #[row_idx, col_idx]
        all_L_tmp = outputs["encoder_hidden_states"]  # N_layer, bs, seq_len, dim
        # print(all_L_tmp[-1][5])
        all_L_tmp = all_L_tmp + outputs["decoder_hidden_states"]  # decoder =h_t-1
        all_L_tmp = all_L_tmp + tuple([x[:, 1:] for x in outputs["decoder_hidden_states"]])  # h_t
        # print(all_L_tmp[-1][5])

    elif model.model_name in ["gpt2"]:
        outputs = model.transformer(input_ids=input_ids,
                                    attention_mask=attn,
                                    output_hidden_states=True,
                                    use_cache=True)
        last_layer_tmp = outputs[0]
        ffn_tmp = last_layer_tmp #[row_idx, col_idx]  # bs, n_phrase, dim
        all_L_tmp = outputs["hidden_states"]  # N_layer, bs, seq_len, dim
        dummpy = torch.zeros((last_layer_tmp.size(0), 1, last_layer_tmp.size(0)))
        all_L_tmp = all_L_tmp + tuple([torch.cat([dummpy, x], dim=1) for x in outputs["hidden_states"]])  # h_t-1
    else:
        raise Exception
    # ffn_tmp = ffn_tmp[row_idx, col_idx]
    return all_L_tmp, ffn_tmp

def Encode_LM(tokenizer, model, sent_list, mask_col_idx,
                    max_tokens=8192, layers = None):

    sorted_idx = np.argsort([-1 * len(x) for x in sent_list])
    original_idx = np.argsort(sorted_idx)
    input_ids_tmp = []
    phrase_all_states = []
    # phrase_ffn = []
    max_len = None
    idx_list_tmp=[]
    for i in sorted_idx: #sort by sent len
        sent = sent_list[i].copy()
        idx_list_tmp.append(i)
        if max_len == None:
            max_len = len(sent)
        # print(sent)
        assert isinstance(sent, list)
        # assert len(sent) <= max_len
        input_ids_tmp.append(sent + [tokenizer.pad_token_id] * (max_len - len(sent)))
        if len(input_ids_tmp) * max_len>=max_tokens or i == sorted_idx[-1]:
            col_idx = mask_col_idx[idx_list_tmp] #bs, idx
            row_idx = np.arange(len(col_idx))[:, None]
            if torch.cuda.is_available():
                input_ids_tmp = torch.cuda.LongTensor(input_ids_tmp)
            else:
                input_ids_tmp = torch.LongTensor(input_ids_tmp)
            N_sent, N_token = input_ids_tmp.size()
            device = input_ids_tmp.device
            token_tmp = torch.LongTensor([0] * (N_sent * N_token)).view(N_sent, N_token).to(device)
            attn_tmp = input_ids_tmp != tokenizer.pad_token_id
            attn_tmp = attn_tmp.long()
            all_L_tmp, _ = \
                _Encode_model(model, input_ids_tmp, token_tmp,attn_tmp,col_idx)
            assert layers is not None
            if model.model_name == "xlnet":
                phrase_all_states_tmp = torch.stack([all_L_tmp[k][row_idx, col_idx] if k%2==0 else all_L_tmp[k] for k in layers])
            else:
                phrase_all_states_tmp = torch.stack([all_L_tmp[k][row_idx, col_idx].detach() for k in layers])
            phrase_all_states.append(phrase_all_states_tmp)
            max_len = None
            input_ids_tmp = []
            idx_list_tmp = []

    # phrase_ffn = torch.cat(phrase_ffn, dim=0) #bs, n_mask, dim
    phrase_all_states = torch.cat(phrase_all_states, dim=1) #N_layer, bs, n_mask, dim
    # phrase_ffn = phrase_ffn[original_idx]
    phrase_all_states = phrase_all_states[:, original_idx]

    return phrase_all_states

def Get_model(model_path, is_cuda):
    if is_cuda:
        device = "cuda"
    else:
        device = "cpu"
    if model_path == "camembert-base":
        model = CamembertForMaskedLM.from_pretrained(model_path)
        model.to(device)
        model.Output_layer = model.lm_head.decoder.weight.data
        model.model_name ="roberta"
    elif model_path.startswith("princeton-nlp/sup-simcse-bert-large-uncased"):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = BertForMaskedLM.from_pretrained(model_path)
        model.model_name = "bert"
        model.to(device)
        model.Output_layer = model.cls.predictions.decoder.weight.data
        model.Output_layer_bias = model.cls.predictions.bias.data
    elif model_path.startswith("dbmdz/bert"):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = BertForMaskedLM.from_pretrained(model_path)
        model.model_name = "bert"
        model.to(device)
        model.Output_layer = model.cls.predictions.decoder.weight.data
        model.Output_layer_bias = model.cls.predictions.bias.data
    elif model_path.startswith("microsoft/mpnet"):
        tokenizer = MPNetTokenizer.from_pretrained(model_path)
        model = MPNetForMaskedLM.from_pretrained(model_path)
        model.to(device)
        model.Output_layer = model.lm_head.decoder.weight.data
        model.Output_layer_bias = model.lm_head.bias.data
        model.model_name ="mpnet"
    elif model_path.startswith("sentence-transformers"):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        model.to(device)
        model.Output_layer = None
        model.Output_layer_bias = None
        # model.Output_layer = model.lm_head.weight.data
        # model.Output_layer_bias = torch.zeros(len(model.Output_layer))
        if model_path=="sentence-transformers/all-roberta-large-v1":
            model.model_name = "sbert-roberta"
        elif model_path in ["sentence-transformers/bert-large-nli-stsb-mean-tokens", "sentence-transformers/bert-large-nli-mean-tokens"]:
            model.model_name = "sbert-bert"
        elif model_path in ["sentence-transformers/all-mpnet-base-v2","sentence-transformers/paraphrase-mpnet-base-v2"]:
            model.model_name = "sbert-mpnet"

    elif model_path.startswith("google/electra") or model_path.startswith("dbmdz/electra"):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = ElectraForPreTraining.from_pretrained(model_path)
        model.to(device)
        model.Output_layer_bias = None
        model.Output_layer = None
        model.model_name = "electra"

    elif model_path == 'facebook/m2m100_418M':
        model = M2M100ForConditionalGeneration.from_pretrained(model_path)
        tokenizer = M2M100Tokenizer.from_pretrained(model_path, src_lang="en",
                                                    tgt_lang="en")
        model.eval()
        model.model_name='m2m100_418M'
    elif model_path.startswith("gpt2"):
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        tokenizer.add_tokens(["<pad>"], special_tokens=True)
        tokenizer.add_tokens(["<mask>"], special_tokens=True)
        tokenizer.pad_token = '<pad>'
        tokenizer.mask_token = '<mask>'
        model = GPT2LMHeadModel.from_pretrained(model_path)
        model.to(device)
        model.Output_layer = model.lm_head.weight.data
        model.Output_layer_bias = torch.zeros(len(model.Output_layer))
        model.model_name = "gpt2"
        tokenizer.vocab = tokenizer.get_vocab()
    elif model_path.startswith("xlnet"):
        tokenizer = XLNetTokenizer.from_pretrained(model_path)
        model = XLNetLMHeadModel.from_pretrained(model_path)
        model.to(device)
        model.Output_layer = model.lm_loss.weight.data
        model.Output_layer_bias = model.lm_loss.bias.data
        model.model_name = "xlnet"
        tokenizer.vocab = tokenizer.get_vocab()
    elif model_path.startswith("t5"):
        space_as_token = False
        if model_path.endswith("_enc"):
            model_path = model_path[:-4]
            model = T5EncoderModel.from_pretrained(model_path)
        else:
            model = T5ForConditionalGeneration.from_pretrained(model_path)

        tokenizer = T5Tokenizer.from_pretrained(model_path)
        tokenizer.mask_token = "<extra_id_0>"
        tokenizer.mask_token_id = 32099
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        model.to(device)
        model.Output_layer = model.lm_head.weight.data
        model.Output_layer_bias = torch.zeros(len(model.Output_layer))
        model.model_name = "t5"
    elif model_path.startswith('openai-gpt'):
        model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
        model.to(device)
        model.Output_layer = model.lm_head.weight.data
        model.Output_layer_bias = torch.zeros(len(model.Output_layer))
        model.model_name = "gpt-1"
        tokenizer = OpenAIGPTTokenizer.from_pretrained(model_path)

    elif model_path.startswith("spanbert"):
        raise NotImplementedError
        # space_as_token = False
        # #model = SpanBERTForMaskedLM.from_pretrained("../SpanBERT/model/"+model_path)
        # model = SpanBertForMaskedLM_SBO.from_pretrained("../SpanBERT/model/" + model_path)
        # model.to(device)
        # model.model_name = "spanbert"
        # model.Output_layer = model.cls.predictions.decoder.weight.data
        # model.Output_layer_bias = model.cls.predictions.bias.data
        # tokenizer = SpanBertTokenizer.from_pretrained(model_path, do_lower_case=False)
        # tokenizer.mask_token = "[MASK]"
        # tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids(["[MASK]"])[0]
        # tokenizer.pad_token = "[PAD]"
        # tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(["[PAD]"])[0]
        # tokenizer.sep_token = "[SEP]"
        # tokenizer.sep_token_id = tokenizer.convert_tokens_to_ids(["[SEP]"])[0]
        # tokenizer.cls_token = "[CLS]"
        # tokenizer.cls_token_id = tokenizer.convert_tokens_to_ids(["[CLS]"])[0]
        # tokenizer.name_or_path = model_path

    elif model_path.startswith("albert"):
        tokenizer = AlbertTokenizer.from_pretrained(model_path)
        model = AlbertForMaskedLM.from_pretrained(model_path)
        model.model_name = "albert"
        model.to(device)
        model.Output_layer = model.predictions.decoder.weight.data
        model.Output_layer_bias = model.predictions.bias.data
        tokenizer.vocab = tokenizer.get_vocab()
    elif model_path.startswith("bert") or model_path.startswith("neuralmind/bert"):
        space_as_token = False
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = BertForMaskedLM.from_pretrained(model_path)
        model.model_name = "bert"
        model.to(device)
        model.Output_layer = model.cls.predictions.decoder.weight.data
        model.Output_layer_bias = model.cls.predictions.bias.data

    elif model_path.startswith("roberta"):
        space_as_token = True
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = RobertaForMaskedLM.from_pretrained(model_path)
        model.model_name = "roberta"
        model.to(device)
        model.Output_layer = model.lm_head.decoder.weight.data
        model.Output_layer_bias = model.lm_head.decoder.bias.data
    elif model_path.startswith("google/pegasus"):
        tokenizer = PegasusTokenizer.from_pretrained(model_path)
        model = PegasusForConditionalGeneration.from_pretrained(model_path)
        model.model_name = "pegasus"
        model.to(device)
        ####The paper says MLM is not used during training###
        # tokenizer.mask_token = tokenizer.mask_token_sent
        # tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        model.Output_layer = model.lm_head.weight.data
        model.Output_layer_bias = torch.zeros(len(model.Output_layer)).to(device)
    elif model_path.startswith("xlm"):
        model = XLMWithLMHeadModel.from_pretrained(model_path)
        model.to(device)
        model.Output_layer = model.pred_layer.proj.weight.data
        model.model_name = "xlm"
    elif model_path =="microsoft/deberta-v3-large":
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        model.model_name = "deberta-v3"
        model.to(device)
        model.Output_layer = None
        model.Output_layer_bias = None
    elif model_path =="microsoft/deberta-large":
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        model.model_name = "deberta"
        model.to(device)
        model.Output_layer = None
        model.Output_layer_bias = None
    elif model_path.startswith("facebook/bart"):
        space_as_token = True
        tokenizer = BartTokenizer.from_pretrained(model_path)
        model = BartForConditionalGeneration.from_pretrained(model_path)
        model.model_name ="fbbart"
        tokenizer.vocab = tokenizer.get_vocab()
        if model_path == "facebook/bart-base":
            # pass
            bart = torch.hub.load('pytorch/fairseq', 'bart.base')
            tokenid = bart.task.source_dictionary.indices["<mask>"]
            mask_emb = bart.model.encoder.embed_tokens.weight[tokenid].data
            model.model.encoder.embed_tokens.weight[tokenizer.mask_token_id] = mask_emb
        model.to(device)
        model.Output_layer = model.lm_head.weight.data
        model.Output_layer_bias = torch.zeros(len(model.Output_layer)).to(device)
    elif model_path =="facebook/mbart-large-50-one-to-many-mmt":
        tokenizer = MBart50TokenizerFast.from_pretrained(model_path, src_lang="en_XX")
        # tokenizer.src_lang = "en_XX"
        model = MBartForConditionalGeneration.from_pretrained(model_path)
        model.model_name ="fbmbart_MT"
        # tokenizer.tgt_lang_id = tokenizer.convert_tokens_to_ids("en_XX")
        # assert tokenizer.convert_ids_to_tokens(tokenizer.tgt_lang_id)=="en_XX"
        # tokenizer.vocab = tokenizer.get_vocab()
        model.to(device)
        model.Output_layer = None
        model.Output_layer_bias = None
    if model.Output_layer_bias is not None:
        model.Output_layer_bias = model.Output_layer_bias.view(-1, 1).to(device)
    model.eval()
    if model.model_name == "spanbert":
        pass
    elif model.model_name in ["gpt2", "xlnet","pegasus",'marian']:
        you_id = tokenizer('you', add_special_tokens=False)['input_ids']
        assert len(you_id) == 1
        assert tokenizer.convert_ids_to_tokens(you_id) == tokenizer.tokenize('you')
    else:
        you_id = tokenizer('you')['input_ids'][1:-1]
        assert len(you_id)==1
        assert tokenizer.convert_ids_to_tokens(you_id)==tokenizer.tokenize('you')
    return model, tokenizer
