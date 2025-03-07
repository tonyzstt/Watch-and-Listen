python train.py \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --meta_file_path /home/tonyzst/Desktop/CS229-Project/data/MMTrail/train/metas_video_convs.json \
    --data_folder /home/tonyzst/Desktop/CS229-Project/data/MMTrail/train \
    --output_dir /home/tonyzst/Desktop/CS229-Project/ \
    --is_multimodal \
    --has_video \
    --has_image \
    --has_audio \