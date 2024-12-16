import csv
import pandas as pd
import torch
from empathy_classifier import EmpathyClassifier



'''
Example:
'''

def predict_epitome_values():
	if torch.cuda.is_available():
		device = torch.device("cuda")
	else:
		print('No GPU available, using the CPU instead.')
		device = torch.device("cpu")

	input_path = 'dataset/sample_test_input.csv'
	output_path = 'reddit_test_output.csv'

	input_df = pd.read_csv(input_path, header=0)

	ids = input_df.id.astype(str).tolist()
	seeker_posts = input_df.seeker_post.astype(str).tolist()
	response_posts = input_df.response_post.astype(str).tolist()


	empathy_classifier = EmpathyClassifier(device,
							ER_model_path = 'trained_models/reddit_ER.pth', 
							IP_model_path = 'trained_models/reddit_IP.pth',
							EX_model_path = 'trained_models/reddit_EX.pth')


	output_file = codecs.open(output_path, 'w', 'utf-8')
	print(seeker_posts)
	print(response_posts)
	print('ayyy lmao')
	epitome_df = pd.DataFrame(columns=['ids', 'seeker_posts', 'response_posts', 'predictions_ER', 'predictions_IP', 'predictions_EX', 'predictions_rationale_ER', 'predictions_rationale_IP', 'predictions_rationale_EX'])
	for i in range(len(seeker_posts)):
		(logits_empathy_ER, predictions_ER, logits_empathy_IP, predictions_IP, logits_empathy_EX, predictions_EX, logits_rationale_ER, predictions_rationale_ER, logits_rationale_IP, predictions_rationale_IP, logits_rationale_EX,predictions_rationale_EX) = empathy_classifier.predict_empathy([seeker_posts[i]], [response_posts[i]])
		epitome_df.loc[0] = [ids[i], seeker_posts[i], response_posts[i], predictions_ER[0], predictions_IP[0], predictions_EX[0], predictions_rationale_ER[0].tolist(), predictions_rationale_IP[0].tolist(), predictions_rationale_EX[0].tolist()]
	print(epitome_df)

	return 0

#parser = argparse.ArgumentParser("Test")
#parser.add_argument("--output_path", type=str, help="output file path")

#parser.add_argument("--ER_model_path", type=str, help="path to ER model")
#parser.add_argument("--IP_model_path", type=str, help="path to IP model")
#parser.add_argument("--EX_model_path", type=str, help="path to EX model")

#args = parser.parse_args()


'''
csv_writer = csv.writer(output_file, delimiter=',', quotechar='"')

csv_writer.writerow(['id','seeker_post','response_post','ER_label','IP_label','EX_label', 'ER_rationale', 'IP_rationale', 'EX_rationale'])

for i in range(len(seeker_posts)):
	(logits_empathy_ER, predictions_ER, logits_empathy_IP, predictions_IP, logits_empathy_EX, predictions_EX, logits_rationale_ER, predictions_rationale_ER, logits_rationale_IP, predictions_rationale_IP, logits_rationale_EX,predictions_rationale_EX) = empathy_classifier.predict_empathy([seeker_posts[i]], [response_posts[i]])

	csv_writer.writerow([ids[i], seeker_posts[i], response_posts[i], predictions_ER[0], predictions_IP[0], predictions_EX[0], predictions_rationale_ER[0].tolist(), predictions_rationale_IP[0].tolist(), predictions_rationale_EX[0].tolist()])

output_file.close()
'''

predict_epitome_values()