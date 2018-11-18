cat vocab_preprocessed.txt | sed "s/^\s\+//g" | sort -rn | grep -v "^[1234]\s" | cut -d' ' -f2 > vocab_cut_preprocessed.txt
