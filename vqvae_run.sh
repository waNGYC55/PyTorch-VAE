## download LJSpeech, then run the following
python vqvae_data_preparation.py --LJSpeech_dir C:/Users/YC/Desktop/TTS/data/LJSpeech-1.1 --save_dir ./data

## prepare feature
python vqvae_feat.py --data_dir ./data --save_dir ./feats

## prepare index
python vqvae_index.py --feat_dir ./data --save_dir ./index

## train model
python vqvae_train.py --index_dir ./index --save_dir ./outputs --num_epoch 100 --batch_size 64 --resnet_depth 6 --nj 4
