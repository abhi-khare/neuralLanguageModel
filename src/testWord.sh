#!/bin/sh
for size in 50 100 300 500
do
  python train_words.py $size chinese5k >> words_log_$size.txt
done
