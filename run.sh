deepspeed --include localhost:$1 --master_port $2 copyright_attack.py \
    --deepspeed --seed_prompt_model $3 --seed_inst_optim_llm $4 --model $5 --generation_blackbox $6 \
    --rephrase_iter $8 --seed_automated --unchange_update_num $7 --logging --file_suffix $9 --test_type $10 