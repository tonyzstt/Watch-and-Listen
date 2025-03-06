# Usage

## Dataset module

After you run `python preprocess.py` under `preprocess/` directory, you can obtain the required metadata json files (e.g. `metas_audio_convs.json`). 

You need to change the following four arguments in `run.sh`.

```bash
--model_name_or_path $PATH_TO_LLM_CKP$/vicuna-7b-v1.5 \
--meta_file_path $PATH_TO_META_DATA_JSON_FILE$/metas_video_convs.json \
--data_folder $ROOT_DIR_OF_MMTRAIL_DATASET$/train/ \
--output_dir $PATH_TO_STORE_TRAINING_OUTPUT$ \
```

Then, by running `bash run.sh` under `src/` directory, you can see a sample data in the dataset.