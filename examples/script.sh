python  train_lm.py --evaluation_strategy steps --per_device_train_batch_size 24 \
         --per_device_eval_batch_size 24 --gradient_accumulation_steps 4 --eval_accumulation_steps 4 \
         --save_strategy steps --save_steps 100 --eval_steps 100 --logging_steps 100 \
         --dataloader_num_workers 2 --torch_compile True --output_dir ./out \
         --jit_mode_eval True --learning_rate 2e-5 --warmup_ratio 0.1 \
         --dataset_name ZurabDz/geo_small_corpus_dedublicated_trash_off_tokenized \
         --test_size 0.05 --tokenizer_name ZurabDz/GeoSentencePieceBPE_32768_v2 \
         --model_name albert-base-v2 --from_scratch --fp16 True \
         --num_train_epochs 64