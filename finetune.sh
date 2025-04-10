python train.py --datas train.csv \
                --batch-size 32 \
                --valid-ratio 0 \
                --no-resume \
                --ckpt-path ./ckpt/pretrained.ckpt \
                --epoch 20 \
                --finetune \
                --eval \
                --model NamBert \
                --workers 0 \
                --test-data sighan_2013_test.csv,sighan_2014_test.csv,sighan_2015_test.csv \
                --ckpt-dir ./ckpt/

