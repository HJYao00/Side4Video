DATA_PATH=/xx/VATEX
python -m torch.distributed.launch --nproc_per_node=8 --master_port 2963 \
main_task_retrieval.py --do_eval --num_thread_reader=4 \
--epochs=5 --batch_size=32 --n_display=50 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/VATEX_Frames \
--output_dir ckpts_vatex/ckpt_msrvtt_retrieval_looseType \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 8 --side_dim 320 \
--datatype vatex \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--pretrained_clip_name ViT-L/14 \
--init_model /xx \
--interaction wti 