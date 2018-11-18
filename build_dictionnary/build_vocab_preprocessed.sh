cat train_pos_preprocessed.txt train_neg_preprocessed.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > vocab_preprocessed.txt
