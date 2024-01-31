DATA_PATH=/xx/MSRVTT
python -m torch.distributed.launch --nproc_per_node=8 --master_port 2234 \
main_task_retrieval.py --do_train --num_thread_reader=4 \
--epochs=5 --batch_size=128 --n_display=50 \
--train_csv ${DATA_PATH}/msrvtt_data/MSRVTT_train.9k.csv \
--val_csv ${DATA_PATH}/msrvtt_data/MSRVTT_JSFUSION_test.csv \
--data_path ${DATA_PATH}/msrvtt_data/MSRVTT_data.json \
--features_path ${DATA_PATH}/frames_30fps \
--output_dir ckpts/side4video_msrvtt \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 --side_dim 320 \
--datatype msrvtt --expand_msrvtt_sentences \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--pretrained_clip_name ViT-B/16 \
--warmup_proportion 0.1 \
--strategy 3 \
--freeze_vit_encoder \
--interaction wti 