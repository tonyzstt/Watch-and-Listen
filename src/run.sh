python dataset.py \
    --model_name_or_path /home/saberwu2002/CS229-Project/hf_ckp/vicuna-7b-v1.5 \
    --meta_file_path /home/saberwu2002/CS229-Project/local_data/MMTrail_processed/test/metas_video_convs.json \
    --data_folder /home/saberwu2002/CS229-Project/local_data/MMTrail_processed/test/ \
    --output_dir /home/saberwu2002/CS229-Project/output/ \
    --is_multimodal \
    --has_video \
    --has_image \
    --has_audio