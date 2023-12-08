#!/usr/bin/env bash

# cd data/sample_path/
# $1 is the input folder containing sample_path.txt 
# usage: bash split_bash_files.sh data/concepnet

cd $1 
echo "changing dir to $1"

filename=sample_path.txt
filename_shuf=sample_path.txt.shuffle

echo "shuffling $1 to $filename_shuf"
shuf $filename -o $filename_shuf
split -l $[ $(wc -l $filename_shuf|cut -d" " -f1) * 90 / 100 ] $filename_shuf train.txt_ --verbose

echo "generating train.txt"
mv train.txt_aa train.txt -v

echo " "
echo "generating dev and test.txt"
filename_eval=train.txt_ab
split -l $[ $(wc -l $filename_eval|cut -d" " -f1) * 50 / 100 ] $filename_eval dev.txt_ --verbose

mv dev.txt_aa dev.txt  -v

mv dev.txt_ab test.txt -v

echo " "
rm $filename_eval -v
remain=dev.txt_ac
if test -f "$remain"; then
    echo "$remain exists."
    rm $remain -v
fi
rm $filename_shuf -v

echo " "
echo "wc -l *"
wc -l *
