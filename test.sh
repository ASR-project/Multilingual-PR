# Testing script to reproduce our test experiments

# python main.py --train False --language nl --subset nl  --network_name Hubert --best_model_run Hubert_nl_tf_freezed
# python main.py --train False --language nl --subset nl  --network_name WavLM --best_model_run WavLM_nl_tf_freezed
# python main.py --train False --language nl --subset nl  --network_name Wav2Vec2 --best_model_run Wav2Vec2_nl_tf_freezed

# python main.py --train False --language sv --subset sv-SE  --network_name Hubert --best_model_run Hubert_sv_tf_freezed
# python main.py --train False --language sv --subset sv-SE  --network_name WavLM --best_model_run WavLM_sv_tf_freezed
# python main.py --train False --language sv --subset sv-SE  --network_name Wav2Vec2 --best_model_run Wav2Vec2_sv_tf_freezed

# python main.py --train False --language it --subset it  --network_name Hubert --best_model_run Hubert_it_tf_freezed
# python main.py --train False --language it --subset it  --network_name WavLM --best_model_run WavLM_it_tf_freezed
# python main.py --train False --language it --subset it  --network_name Wav2Vec2 --best_model_run Wav2Vec2_it_tf_freezed

# python main.py --train False --language ru --subset ru  --network_name Hubert --best_model_run Hubert_ru_tf_freezed
# python main.py --train False --language ru --subset ru  --network_name WavLM --best_model_run WavLM_ru_tf_freezed
# python main.py --train False --language ru --subset ru  --network_name Wav2Vec2 --best_model_run Wav2Vec2_ru_tf_freezed

# python main.py --train False --language tr --subset tr  --network_name Hubert --best_model_run Hubert_tr_tf_freezed
# python main.py --train False --language tr --subset tr  --network_name WavLM --best_model_run WavLM_tr_tf_freezed
# python main.py --train False --language tr --subset tr  --network_name Wav2Vec2 --best_model_run Wav2Vec2_tr_tf_freezed

python main.py --train False --language sv --subset sv-SE  --network_name Hubert --best_model_run Hubert_sv_0.05_train_tf_freezed --limit_train_batches 0.05
python main.py --train False --language sv --subset sv-SE  --network_name WavLM --best_model_run WavLM_sv_0.05_train_tf_freezed --limit_train_batches 0.05
python main.py --train False --language sv --subset sv-SE  --network_name Wav2Vec2 --best_model_run Wav2Vec2_sv_0.05_train_tf_freezed --limit_train_batches 0.05

python main.py --train False --language sv --subset sv-SE  --network_name Hubert --best_model_run Hubert_sv_0.1_train_tf_freezed --limit_train_batches 0.1
python main.py --train False --language sv --subset sv-SE  --network_name WavLM --best_model_run WavLM_sv_0.1_train_tf_freezed --limit_train_batches 0.1
python main.py --train False --language sv --subset sv-SE  --network_name Wav2Vec2 --best_model_run Wav2Vec2_sv_0.1_train_tf_freezed --limit_train_batches 0.1

python main.py --train False --language sv --subset sv-SE  --network_name Hubert --best_model_run Hubert_sv_0.5_train_tf_freezed --limit_train_batches 0.5
python main.py --train False --language sv --subset sv-SE  --network_name WavLM --best_model_run WavLM_sv_0.5_train_tf_freezed --limit_train_batches 0.5
python main.py --train False --language sv --subset sv-SE  --network_name Wav2Vec2 --best_model_run Wav2Vec2_sv_0.5_train_tf_freezed --limit_train_batches 0.5