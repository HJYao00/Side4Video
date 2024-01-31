DATA_PATH=/xx/MSVD
python -m torch.distributed.launch --nproc_per_node=8 --master_port 12963 \
main_task_retrieval.py --do_eval --num_thread_reader=4 \
--epochs=5 --batch_size=16 --n_display=50 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/MSVD_frames \
--output_dir ckpts_msvd/ckpt_msrvtt_retrieval_looseType \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 8 --side_dim 320 \
--datatype msvd \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--pretrained_clip_name EVA02-CLIP-bigE-14 \
--init_model /xx \
--interaction wti 