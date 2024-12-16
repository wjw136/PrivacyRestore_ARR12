cd ../Training

# test mc scores
CUDA_VISIBLE_DEVICES=0 python ./test_mc.py \
                        --model_name_or_path "" \
                        --bank_interventions_dir "" \
                        --submodule_path "" \
                        --data_path "" \
                        --auxiliary_model_name_or_path "" \
                        --inter_heads_path "" \
                        --evidences_path "" \
                        --seed 50


# test generation score(LLM-J or ROUGE-L)
export OPENAI_API_BASE=""
export OPENAI_API_KEY=""
CUDA_VISIBLE_DEVICES=0 python ./test_gen.py \
                        --model_name_or_path "" \
                        --bank_interventions_dir "" \
                        --submodule_path "" \
                        --data_path "" \
                        --auxiliary_model_name_or_path "" \
                        --inter_heads_path "" \
                        --evidences_path "" \
                        --output_ans_dir "" \
                        --seed 48

# test speed (throughput)
CUDA_VISIBLE_DEVICES=0 python ./test_speed.py \
                        --model_name_or_path "" \
                        --bank_interventions_dir "" \
                        --submodule_path '' \
                        --data_path "" \
                        --auxiliary_model_name_or_path "" \
                        --inter_heads_path "" \
                        --evidences_path '' \
                        --output_ans_dir '' \
                        --seed 48