## download LJSpeech, then run the following
python vqvae_data_preparation.py --LJSpeech_dir C:/Users/YC/Desktop/TTS/data/LJSpeech-1.1 --save_dir ./data

## prepare feature
python vqvae_feat.py --data_dir ./data --save_dir ./feats

## train model
python vqvae_train.py  --data_dir ./data --save_dir ./outputs --num_epoch 50 --batch_size 1 --resnet_depth 6 --nj 1
