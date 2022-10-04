# lexsub_decontextualised
Code for "Unsupervised Lexical Substitution with Decontextualised Embeddings" (COLING 2022)

Decontextualised embeddings used in our English and Italian experiments are available at [this Bitbucket repository](https://bitbucket.org/TakashiW/lexical_substitution/src/main).

## Replicate Experiments on SWORDS (Using DeBERTa-V3)
### Generation Performance

1. Download decontextualised embeddings "deberta-v3-large.tar.bz2" from [this Bitbucket repository](https://bitbucket.org/TakashiW/lexical_substitution/src/main) and decompress the folder, e.g. using the command "tar -xf deberta-v3-large.tar.bz2".

2. Clone the [SWORDS data set](https://github.com/p-lambda/swords)

3. Prepare input files using the following command:

```
python swords_preprocess.py -folder path_to_SWORDS -data_split test
```
**This will produce the files named "swords_masked_sent_test.txt" and "swords_substitute_cands_list_test.pkl"**. In the first file, each line contains a target word and its context (seperated by a tab space "\t"), where the target word is replaced with \<mask\>. (e.g. "substitution\tWe propose a new method for lexical \<mask\> using pre-trained language models."). The second file contains a list of gold candidates (used for the ranking task).


4. Generate substitute candidates using the following command:

```
folder=Result
tgt_sent=swords_masked_sent_test.txt
model=microsoft/deberta-v3-large
model_basename=$(basename "$model")
vec="${model_basename}/K4/K0/vec3-22.txt ${model_basename}/K4/K1/vec3-22.txt ${model_basename}/K4/K2/vec3-22.txt ${model_basename}/K4/K3/vec3-22.txt"
CUDA_VISIBLE_DEVICES=0 python generate.py -folder ${folder} -model ${model} -vec ${vec}  -tgt_sent ${tgt_sent} -lev 0.5 -beam_size 50
```
This will produce the file "microsoft_deberta-v3-large_beam_50lambda_val0.7_candidates2cossim_score.pkl" in the "Result" folder. The "-lev" option denotes the threshold for the edit-distance heuristic (set "-lev 0" to disable this), and "-beam_size" denotes the number of candidates to generate.

5. Rerank the candidates using the following command:

```
folder=Result
model=microsoft/deberta-v3-large
candidates=Result/microsoft_deberta-v3-large_beam_50lambda_val0.7_candidates2cossim_score.pkl
tgt_sent=swords_masked_sent_test.txt
CUDA_VISIBLE_DEVICES=0 python reranking.py -candidates ${candidates} -folder ${folder} -model ${model} -tgt_sent ${tgt_sent}
```
This will produce the file "microsoft_deberta-v3-large_candidates2reranking_score.pkl" in the "Result" folder. 

6. Prepare files for the SWORDS evaluation scrpit using the following command:

```
python swords_postprocess.py -i Result/microsoft_deberta-v3-large_candidates2reranking_score.pkl -swords_path path_to_SWORDS -save_name path_to_SWORDS/notebooks/swords-v1.1_test_mygenerator.lsr.json
```

(If you want to evaluate the performance without reranking, set "-i" to "Result/microsoft_deberta-v3-large_beam_50lambda_val0.7_candidates2cossim_score.pkl" intead.)

7. Calculate F scores using the evaluation script on [SWORDS](https://github.com/p-lambda/swords):

```
./cli.sh eval swords-v1.1_test --result_json_fp notebooks/swords-v1.1_test_mygenerator.lsr.json --output_metrics_json_fp notebooks/mygenerator.metrics.json
```

The result should be "33.61, 65.84, 24.52, 39.90 (lenient_a_f@10, lenient_c_f@10, strict_a_f@10, strict_c_f@10)" — the scores shown in Table 1. To evaluate another model, e.g. BERT, you can replace "model=microsoft/deberta-v3-large" with "model=bert-large-uncased" in the generation and reranking steps. 

### Ranking Performance

Replace "candidates=Result/microsoft_deberta-v3-large_beam_50lambda_val0.7_candidates2cossim_score.pkl" with **"candidates=swords_substitute_cands_list_test.pkl"** in Step 5 above, and then prepare files for evaluation as in 6. Then, calculate the GAP score using the evaluation script on [SWORDS](https://github.com/p-lambda/swords):

```
./cli.sh eval swords-v1.1_test --result_json_fp notebooks/swords-v1.1_test_mygenerator.lsr.json --output_metrics_json_fp notebooks/mygenerator.metrics.json --metrics gap_rat
```
The result should be "62.9" — the score shown in Table 2.

## Replicate Experiments on SemEval-07 (Using DeBERTa-V3)

1. Download SemEval-07 data ("All Gold Standard and Scoring Data") at [http://www.dianamccarthy.co.uk/task10index.html](http://www.dianamccarthy.co.uk/task10index.html)

2. Prepare input files using the following command:

```
python SemEval07_preprocess.py -folder path_to_SemEval_data
```
**This will produce "SemEval07_masked_sent.txt", "SemEval07_candidates.pkl" and SemEval07_tgtids.txt".**

3. Rerank the candidates using the following command:

```
folder=Result_SemEval
model=microsoft/deberta-v3-large
candidates=SemEval07_candidates.pkl
tgt_sent=SemEval07_masked_sent.txt
CUDA_VISIBLE_DEVICES=0 python reranking.py -candidates ${candidates} -folder ${folder} -model ${model} -tgt_sent ${tgt_sent}
```
This will produce the file "microsoft_deberta-v3-large_candidates2reranking_score.pkl" in the "Result_SemEval" folder. 

4. Prepare a file for evaluation using the following command:

```
out=save_file_path
python SemEval07_postprocess.py -file Result_SemEval/microsoft_deberta-v3-large_candidates2reranking_score.pkl -out ${out}
```

5. Clone [this GitHub repository](https://github.com/orenmel/lexsub), and run its following evaluation script:

```
python jcs/evaluation/lst/lst_gap.py datasets/lst_all.gold ${out} result.txt no-mwe
tail -n 1 result.txt
```

## Replicate Italian Experiments (Using ELECTRA)

1. Download decontextualised embeddings "electra-base-italian-xxl-cased-discriminator.tar.bz2" from [this Bitbucket repository](https://bitbucket.org/TakashiW/lexical_substitution/src/main).

2. Download test data at the [EVALITA 2009 workshop page](https://www.evalita.it/campaigns/evalita-2009/tasks/lexical-substitution/)

3. Prepare input files using the following command (**This code requires the spaCy version be "3.2.2"**):
```
python EVALITA_preprocess.py -folder path_to_EVALITA_data 
```
This will produce the files "Italian_masked_sent.txt", "Italian_gold.pkl", and "tgt_lemma_pos_list.txt".

4. Generate substitute candidates using the following command:

```
folder=Result_Italian
tgt_sent=Italian_masked_sent.txt
model=dbmdz/electra-base-italian-xxl-cased-discriminator
model_basename=$(basename "$model")
vec="${model_basename}/K4/K0/vec3-10.txt ${model_basename}/K4/K1/vec3-10.txt ${model_basename}/K4/K2/vec3-10.txt ${model_basename}/K4/K3/vec3-10.txt"
CUDA_VISIBLE_DEVICES=0 python generate.py -folder ${folder} -model ${model} -vec ${vec}  -tgt_sent ${tgt_sent} -lev 0.5 -beam_size 50
```
This will produce the file "electra-base-italian-xxl-cased-discriminator_beam_50lambda_val0.7_candidates2cossim_score.pkl" in the "Result_Italian" folder. 

5. Rerank the candidates using the following command:

```
folder=Result_Italian
model=dbmdz/electra-base-italian-xxl-cased-discriminator
candidates=Result_Italian/electra-base-italian-xxl-cased-discriminator_beam_50lambda_val0.7_candidates2cossim_score.pkl
tgt_sent=Italian_masked_sent.txt
CUDA_VISIBLE_DEVICES=0 python reranking.py -candidates ${candidates} -folder ${folder} -model ${model} -tgt_sent ${tgt_sent}
```
This will produce the file "electra-base-italian-xxl-cased-discriminator_candidates2reranking_score.pkl" in the "Result_Italian" folder. 

6. Prepare files for the EVALITA-2009 evaluation script using the following command:
(**This code requires the spaCy version be "3.2.2"**)
```
python EVALITA_postprocess.py -i Result_Italian/electra-base-italian-xxl-cased-discriminator_candidates2reranking_score.pkl -gold path_to_EVALITA_data/gold.test
```
This will show F scores (as well as precision and recall) and the result is save as "Result_Italian/electra-base-italian-xxl-cased-discriminator_candidates2reranking_score_scores.txt". **It will also produce "Result_Italian/electra-base-italian-xxl-cased-discriminator_candidates2reranking_score_candidates-oot.txt" and "Result_Italian/electra-base-italian-xxl-cased-discriminator_candidates2reranking_score_candidates-best.txt" in the "Result_Italian" folder.** These files are used as inputs for the EVALITA-2009 evaluation script.


7. Calculate "best" and "oot" scores using the following commands:
```
folder=path_to_EVALITA_data
perl ${folder}/score.pl Result_Italian/electra-base-italian-xxl-cased-discriminator_candidates2reranking_score_candidates-oot.txt ${folder}/gold.test -t oot
perl ${folder}/score.pl Result_Italian/electra-base-italian-xxl-cased-discriminator_candidates2reranking_score_candidates-best.txt ${folder}/gold.test -t best
```
