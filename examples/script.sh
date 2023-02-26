python  train_lm.py --evaluation_strategy steps --per_device_train_batch_size 24 \
         --per_device_eval_batch_size 24 --gradient_accumulation_steps 4 --eval_accumulation_steps 4 \
         --save_strategy steps --save_steps 100 --eval_steps 100 --logging_steps 100 \
         --dataloader_num_workers 2 --torch_compile False --output_dir ./