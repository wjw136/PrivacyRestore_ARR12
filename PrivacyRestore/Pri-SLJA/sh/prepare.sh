cd ../bankConstruction

model_size="7b"
model_path=""
num_to_intervene=175

CUDA_VISIBLE_DEVICES=1 python get_activations.py med \
        --model_dir "$model_path" \
        --output_activation_dir "" \
        --data_type "question" \
        --train_data_path ""

CUDA_VISIBLE_DEVICES=1 python bankConstruction.py \
            --model_dir "$model_path" \
            --activations_dir "" \
            --bank_dir "" \
            --mode med \
            --num_to_intervene "$num_to_intervene"

CUDA_VISIBLE_DEVICES=1 python bankTopHeads.py \
    --bank_dir "" \
    --mask_symptoms_path "" \
    --output_dir "" \
    --model_dir "$model_path" \
    --num_heads "$num_to_intervene" \
    --mode "agg"
