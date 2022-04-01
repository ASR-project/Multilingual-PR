python main.py --train True --language nl --subset nl  --network_name WavLM    --lr 3e-3 
python main.py --train True --language sv --subset sv-SE  --network_name WavLM --lr 3e-3
# python main.py --train True --language tr --subset tr  --network_name WavLM    --lr 2e-2
# python main.py --train True --language it --subset it  --network_name WavLM    --lr 2e-3
# python main.py --train True --language ru --subset ru  --network_name WavLM 

python main.py --train False --language nl --subset nl  --network_name WavLM --best_model_run WavLM_nl_tf_freezed
python main.py --train False --language sv --subset sv-SE  --network_name WavLM --best_model_run WavLM_sv_tf_freezed
python main.py --train False --language it --subset it  --network_name WavLM --best_model_run WavLM_it_tf_freezed
# python main.py --train False --language ru --subset ru  --network_name WavLM --best_model_run WavLM_ru_tf_freezed
python main.py --train False --language tr --subset tr  --network_name WavLM --best_model_run WavLM_tr_tf_freezed



# python main.py --train True --language nl --subset nl  --network_name WavLM --freeze_transformer False
# python main.py --train True --language sv --subset sv-SE  --network_name WavLM --freeze_transformer False
# python main.py --train True --language tr --subset tr  --network_name WavLM --freeze_transformer False
# python main.py --train True --language it --subset it  --network_name WavLM --freeze_transformer False
# python main.py --train True --language ru --subset ru  --network_name WavLM --freeze_transformer False



# python main.py --train False --language nl --subset nl  --network_name WavLM --best_model_run WavLM_nl
# python main.py --train False --language sv --subset sv-SE  --network_name WavLM --best_model_run WavLM_sv
# python main.py --train False --language it --subset it  --network_name WavLM --best_model_run WavLM_it
# python main.py --train False --language ru --subset ru  --network_name WavLM --best_model_run WavLM_ru
# python main.py --train False --language tr --subset tr  --network_name WavLM --best_model_run WavLM_tr