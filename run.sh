bash download.sh
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
python simulator.py --split test --num_chats 980 --model_name_or_path final2 --disable_output_dialog --output output_.jsonl
python postprocess.py output_.jsonl output__.jsonl
python truncate.py output__.jsonl output.jsonl