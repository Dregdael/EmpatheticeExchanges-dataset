import codecs
import os
import csv
import re
import numpy as np
import argparse

from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)

parser = argparse.ArgumentParser("process_data")
parser.add_argument("--input_path", type=str, help="path to input data")
parser.add_argument("--output_path", type=str, help="path to output data")
args = parser.parse_args()

input_file = codecs.open(args.input_path, 'r', 'utf-8')
output_file = codecs.open(args.output_path, 'w', 'utf-8')

csv_reader = csv.reader(input_file, delimiter = ',', quotechar='"')
csv_writer = csv.writer(output_file, delimiter = ',',quotechar='"') 

next(csv_reader, None) # skip the header

csv_writer.writerow(["id","seeker_post","response_post","level","rationale_labels","rationale_labels_trimmed","response_post_masked"])

for row in csv_reader:
	# sp_id,rp_id,seeker_post,response_post,level,rationales

	seeker_post = row[2].strip()
	response = row[3].strip()

	response_masked = response


	response_tokenized = tokenizer.decode(tokenizer.encode_plus(response, truncation = True, add_special_tokens = True, max_length = 64, padding = 'max_length')['input_ids'], clean_up_tokenization_spaces=False)

	#print(response_tokenized)
	response_tokenized_non_padded = tokenizer.decode(tokenizer.encode_plus(response, truncation = True, add_special_tokens = True, max_length = 64, padding = False)['input_ids'], clean_up_tokenization_spaces=False)
	#print(response_tokenized_non_padded)
	response_words = tokenizer.tokenize(response_tokenized)
	response_non_padded_words = tokenizer.tokenize(response_tokenized_non_padded)
	#print(response_words)
	#print(response_non_padded_words)

	if len(response_words) != 64:
		continue
	
	#print(len(response))

	response_words_position = np.zeros((len(response),), dtype=np.int32)

	#print(len(response_words_position))

	rationales = row[5].strip().split('|')
	#print(rationales)

	rationale_labels = np.zeros((len(response_words),), dtype=np.int32)
	#print(rationale_labels)

	curr_position = 0

	for idx in range(len(response_words)):
		curr_word = response_words[idx]
		if curr_word.startswith('Ġ'):
			curr_word = curr_word[1:]
		response_words_position[curr_position: curr_position+len(curr_word)+1] = idx
		#print(response_words_position)
		curr_position += len(curr_word)+1


	if len(rationales) == 0 or row[5].strip() == '':
		rationale_labels[1:len(response_non_padded_words)] = 1
		#print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
		#print('empty rationale labels')		
		#print(rationale_labels)
		#print(len(rationale_labels))
		response_masked = ''

 
	for r in rationales:
		if r == '':
			continue
		try:
			#print(r)
			r_tokenizer = tokenizer.decode(tokenizer.encode(r, add_special_tokens = False))
			#print(type(r_tokenizer))
			match = re.search(r_tokenizer , response_tokenized)
			#print(match.start(0))
			#print(match)
			curr_match = response_words_position[match.start(0):match.start(0)+len(r_tokenizer)]
			#print(response)
			#print(response_words_position)
			#print(curr_match)
			curr_match = list(set(curr_match))
			#print(curr_match)
			curr_match.sort()
			#print(curr_match)

			response_masked = response_masked.replace(r, ' ')
			response_masked = re.sub(r' +', ' ', response_masked)

			#print(rationale_labels)
			rationale_labels[curr_match] = 1
			#print(rationale_labels)
		except:
			continue
	
	
	#print(f' rationale labels: {rationale_labels}')

	rationale_labels_str = ','.join(str(x) for x in rationale_labels)
	#print(rationale_labels)
	#print(f' rationale labels str: {rationale_labels}')
	rationale_labels_str_trimmed = ','.join(str(x) for x in rationale_labels[1:len(response_non_padded_words)])
	#print(f' rationale labels str trimmed: {rationale_labels}')

	csv_writer.writerow([row[0] + '_' + row[1], seeker_post, response, row[4], rationale_labels_str, len(rationale_labels_str_trimmed), response_masked])
	#exit()

input_file.close()
output_file.close()