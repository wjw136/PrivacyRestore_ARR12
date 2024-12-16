cd ../Training

CUDA_VISIBLE_DEVICES=0 python ./train_llama.py \
                        --model_name_or_path "" \
                        --bank_interventions_dir "" \
                        --data_path "" \
                        --auxiliary_model_name_or_path "" \
                        --inter_heads_path "" \
                        --evidences_path "" \
                        --symptom_antecedent_data_path "" \
                        --level 2 \
                        --epoch 5 \
                        --eval_size 64 \
                        --eval_steps 1024 \
                        --gradients_accumulation_steps 4 \
                        --output_dir ""
