## download LJSpeech, then run the following
python vqvae_data_preparation.py --LJSpeech_dir C:/Users/YC/Desktop/TTS/data/LJSpeech-1.1 --save_dir ./data

## prepare feature
python vqvae_feat.py --data_dir ./data --save_dir ./feats

## prepare index
python vqvae_index.py --feat_dir ./data --save_dir ./index

## train model
python vqvae_train.py --index_dir ./index --save_dir ./outputs --task_name vqvae_batchnorm_linear_tanh --num_epoch 100 --batch_size 128 --resnet_depth 6 --nj 3

## inference
python vqvae_sample.py --index_dir ./index --save_dir ./outputs/samples --model_path ./outputs/vqvae_batchnorm_/models/model_100 --batch_size 1 --nj 1