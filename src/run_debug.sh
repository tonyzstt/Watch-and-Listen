python dataset.py \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --meta_file_path ../data/MMTrail/processed_data/test/metas_video_convs.json \
    --data_folder ../data/MMTrail/processed_data/test/ \
    --output_dir ../output/ \
    --is_multimodal \
    --has_video \
    --has_image \
    --has_audio