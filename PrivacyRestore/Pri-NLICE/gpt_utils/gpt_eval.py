import openai
import os
from typing import Dict, List
import time
from fastchat.model import get_conversation_template
import argparse
import re
import pandas as pd
import json
import datetime
import numpy as np
current_time = datetime.datetime.now()
# 将时间格式化为字符串
time_string = current_time.strftime("%Y-%m-%d-%H:%M:%S")

class GPT():
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_base = os.getenv("OPENAI_API_BASE")
    
    def __init__(self, judge_name) -> None:
        self.model_name = judge_name

    def generate(self, conv: List[Dict], 
                max_n_tokens: int, 
                temperature: float,
                top_p: float = 0.0,):
        '''
        Args:
            conv: List of dictionaries, OpenAI API format
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                response = openai.ChatCompletion.create(
                            model = self.model_name,
                            messages = conv,
                            max_tokens = max_n_tokens,
                            temperature = temperature,
                            top_p = top_p,
                            request_timeout = self.API_TIMEOUT,
                            )
                output = response["choices"][0]["message"]["content"]
                break
            except Exception as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)
        
            time.sleep(self.API_QUERY_SLEEP)
        return output 
    
    def batched_generate(self, 
                        convs_list: List[List[Dict]],
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 0.0,):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]
    



# def get_judge_system_prompt():
#     prompt = f"""You are a helpful medical assistant. You will receive a medical question from a patient, the correct diagosis of this patient and the AI assistant's response to the question. Please act as an impartial judge and evaluate whether the response provided by the AI assistant contains useful information and answers the correct diagosis of the patient. Your evaluation should consider factors including usefulness and accuracy provided in the response. Be as objective as possible. 
# Rate the response on a scale from 1 to 5. Use the scoring rules below to score the model's response to the user question on a scale of 1 to 5:
# Score 1: The model's response is empty or contains no diagnosis, and it doesn't provide any useful information to the patient. This is the most helpless response when the patient consults the AI assistant.
# Score 2: The model's response includes some useful information for the patient, but it provides more than one diagnosis choices or does not provide any diagnosis outcome. While lacking a final diagnosis result, this response is considered superior to Score 1.
# Score 3: The model can generate only one diagnosis outcome for the patient, but unfortunately, it does not correspond to the correct diagnosis. Instead, it provides a random diagnosis, and it even violates the accurate diagnosis result, potentially leading to negative consequences.
# Score 4: The model outputs only one diagnosis outcome that aligns with the symptoms provided by the patient in the question. Although the diagnosis may not match the correct answer, it still offers some benefits for the patient.
# Score 5: The model provides only one diagnosis that is consistent with the correct answer. There are only slight differences in expression between the model's output and the correct answer. This represents the ideal scenario where the AI assistant makes no mistakes.
# Your output should begin with rating and your rating should strictly follow this format: "Rating: [[rating]]", the rating in the double-closed brackets is a number from 1 to 5, e,g, "Rating: [[1]]". After rating, you should give some explanations for you score."""
#     return prompt

def get_judge_system_prompt():
    prompt = "You are a neutral medical evaluator. You will be provided with a medical question from a patient and the AI assistant's response to the question. " + \
        "Your task is to assess whether the AI assistant's response is useful and accurately addresses the diagnosis. Consider factors such as the relevance, clarity, and accuracy of the information provided. " + \
        "Rate the response on a scale from 1 to 10, where 1 indicates poor quality and 10 indicates excellent quality. " + \
        "Start your answer with the rating score by strictly following the format: `[[rating]]`. After assigning a rating, provide a brief explanation for your score, justifying your evaluation."
    return prompt


class GPTJudge():
    def __init__(self, judge_name="gpt-3.5-turbo-1106", judge_max_n_tokens=100, judge_temperature=0.0):
        self.judge_model = GPT(judge_name)
        self.judge_name = judge_name
        self.max_n_tokens = judge_max_n_tokens
        self.temperature = judge_temperature
        self.max_tries = 1

    # def get_judge_prompt(self, question, correct_answer, response):
    #     # response = "..."
    #     # response = "Hypertensive retinopathy."
    #     # response = "Carcinoma."
    #     example2 = "[QUESTION]: 'A 37-year-old female has a history of diabetes, pancreatic cancer in family members, chronic pancreatitis, obesity, smoking. The 37-year-old female presents the symptoms of underweight, pale stools and dark urine, diarrhea, pain, skin lesions, nausea, fatigue, cough. What is the likely diagnosis?'\n[CORRECT DIAGNOSIS]: 'Pancreatic neoplasm.'\n[ASSISTANT RESPONSE]: '(Please select one of the following options)\n\nA) Irritable Bowel Syndrome (IBS)\nB) Inflammatory Bowel Disease (IBD)\nC) Clostridioides.'\nRating: [[2]]. The assistant's response gives more than one possible diagnoses and no final diagnosis."
    #     example1 = "[QUESTION]: 'A 55-year-old female has a history of pancreatic cancer in family members, diabetes, chronic pancreatitis, obesity, smoking. The 55-year-old female presents the symptoms of underweight, pale stools and dark urine, pain, skin lesions, nausea, fatigue. What is the likely diagnosis?'\n[CORRECT DIAGNOSIS]: 'Pancreatic neoplasm.'\n[ASSISTANT RESPONSE]: 'Carcinoma.'\nRating: [[4]]. The assistant's response is not related to the correct diagnosis but aligned with the symptoms."
    #     return f"Here are some examples.\nExample1: \n{example1}\n\nExample2: \n{example2}\n\nPlease strictly follow the format of these examples and give your rating based on the scoring rules firstly followed by some reasons.\n[QUESTION]: '{question}'\n[CORRECT DIAGNOSIS]: '{correct_answer}'\n[ASSISTANT RESPONSE]: '{response}'\nRating: "


    # def get_judge_prompt(self, question, correct_answer, response):
    #     return f"Question: [[{question}]]\nCorrect Diagnosis: [[{correct_answer}]]\nAssitant Response: [[{response}]]\nRating: "

    def get_judge_prompt(self, question, response):
        return f"[User Question]\n{question}\n\n[The Start of Assitant Response]\n{response}\n[The End of Assistant Response]\n\nRating: "


    def create_conv(self, full_prompt):
        conv = get_conversation_template(self.judge_name)
        conv.set_system_message(
            get_judge_system_prompt()
        )
        conv.append_message(conv.roles[0], full_prompt)
        return conv.to_openai_api_messages()

    def process_output(self, raw_output):
        pattern = r'\[\[(\d+)\]\]'
        match = re.search(pattern, raw_output)
        output = int(match.group(1)) if match else None
        if output is None:
            print(f"[Error in processing judge output]: {raw_output}" )
            return 1, False
        return output, True

    def score(self, question_list, correct_answer_list, response_list):
        convs_list = [self.create_conv(self.get_judge_prompt(prompt, correct_answer, response)) for prompt, correct_answer, response in zip(question_list, correct_answer_list, response_list)]
        # print(convs_list[0])

        outputs = []
        for idx, conv in enumerate(convs_list):
            print(f"************[{idx}]/{len(convs_list)}***************", flush=True)
            for tmp_id in range(self.max_tries):
                raw_output = self.judge_model.generate(conv, max_n_tokens = self.max_n_tokens,
                                                            temperature = self.temperature)
                print(conv)
                print(raw_output)

                output, isvalid = self.process_output(raw_output)
                if isvalid:
                    outputs.append(output)
                    break
                if tmp_id == (self.max_tries - 1):
                    outputs.append(1)
            print(f"final output: {outputs[-1]}")

        return outputs
    
    def single_score(self, question, response):
        conv = self.create_conv(self.get_judge_prompt(question, response))
        for tmp_id in range(self.max_tries):
            raw_output = self.judge_model.generate(conv, max_n_tokens = self.max_n_tokens,
                                                        temperature = self.temperature)
            output, isvalid = self.process_output(raw_output)
            if isvalid:
                return output, raw_output
            if tmp_id == (self.max_tries - 1):
                return 1, raw_output


def main():




    parser = argparse.ArgumentParser()

    parser.add_argument('--judge_data_path', type=str, \
                        default='', help='test dataset')
    parser.add_argument('--init_data_path', type=str, \
                        default='', help='test dataset')
    parser.add_argument('--output_data_dir', type=str, \
                        default='', help='test dataset')
    parser.add_argument('--key1', type=str, \
                        default='intervention_gen_str', help='test dataset')
    parser.add_argument('--key2', type=str, \
                        default='correct_str', help='test dataset')
    
    ############ Judge model parameters ##########
    parser.add_argument(
        "--judge-name",
        default="gpt-3.5-turbo-1106",
        help="Name of judge model.",
        choices=["gpt-3.5-turbo-1106", "gpt-4","no-judge", "gpt-3.5-turbo-0125"]
    )
    parser.add_argument(
        "--judge-max-n-tokens",
        type = int,
        default = 100,
        help = "Maximum number of tokens for the judge."
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=0.0,
        help="Temperature to use for judge."
    )


    args = parser.parse_args()
    print(args)

    gptJ = GPTJudge()

    question = "A 46-year-old male has a history of chronic pancreatitis, diabetes, obesity, pancreatic cancer in family members. The 46-year-old male presents the symptoms of cough, diarrhea, nausea, pain, pale stools and dark urine, skin lesions, underweight. What is the likely diagnosis?"
    # correct_answer = "Pancreatic neoplasm"
    response = """Thank you for the question. Based on the symptoms presented, I will provide a likely diagnosis for the patient.

The patient, a 46-year-old male, presents with cough and nausea. The cough could be due to various causes, including:

1. Upper respiratory tract infection (URTI): The most likely cause of cough in this patient population. URTIs are common in adults and can be caused by viruses or bacteria.
2. Bronchitis: A common respiratory condition characterized by inflammation of the bronchial tubes. The patient's cough may be accompanied by chest discomfort, fatigue, and fever.
3. Tuberculosis (TB): A severe and contagious disease that affects the lungs. The patient's cough may be accompanied by weight loss, fever, and night sweats.

The patient's recent illness and fever suggest a likely diagnosis of URTI.

In addition, the patient's recent illness and fever suggest a likely diagnosis of URTI.

Based on the"""
    response2 = """Thank you for the question. Based on the symptoms provided, the likely diagnosis for a 46-year-old male presenting with cough and nausea is a respiratory tract infection, such as bronchitis or pneumonia.

Bronchitis is an inflammation of the bronchial tubes, which can cause a persistent cough and mucus production. Pneumonia, on the other hand, is an infection of the lungs that can cause cough, fever, chills, and difficulty breathing. Both conditions are common and can be caused by viruses or bacteria.

Other possible causes of cough and nausea include gastroesophageal reflux disease (GERD), asthma, and chronic obstructive pulmonary disease (COPD). However, based on the symptoms provided, the most likely diagnosis is a respiratory tract infection.

It is important to note that a thorough medical history and physical examination are necessary to make an accurate diagnosis and determine the appropriate course of treatment. Additional diagnostic tests, such as a chest X-ray or blood work,"""
    response2 = """Based on the symptoms and medical history provided, the likely diagnosis for the 46-year-old male patient is pancreatic cancer.

Pancreatic cancer is a common cause of abdominal pain, weight loss, and gastrointestinal symptoms, including cough, diarrhea, nausea, and pale or dark urine. The patient's history of chronic pancreatitis, diabetes, obesity, and family history of pancreatic cancer also increase the likelihood of this diagnosis.

Pancreatic cancer is the most common cause of cancer-related deaths worldwide, and early detection is critical for improving survival rates. The patient's symptoms, combined with his medical history, suggest that further evaluation and diagnostic testing are warranted to confirm the diagnosis.

Imaging studies, such as CT scans or endoscopic ultrasound, may be used to evaluate the pancreas and detect any tumors or abnormalities. Blood tests, including carcinoembryonic antigen (CEA) and cancer antigen 19.9 (CA19.9), may also be"""

    print(gptJ.single_score(question, response))


    # df = pd.read_excel(args.judge_data_path)
    # data_list = []
    # init_question_list = []
    # with open(args.init_data_path, 'r') as file:
    #     for line in file.readlines():
    #         data_list.append(json.loads(line))
    #         init_question_list.append(data_list[-1]['question_init'])

    # correct_answer_list = list(df[args.key2])
    # response_list = list(df[args.key1])
    # # for item in response_list:
    # #     print(item)
    # #     if item[-1] !=".":
    # #         pass
    # # init_question_list = [ f"{item}." if item[-1]!='.' else item for item in init_question_list]
    # correct_answer_list = [ f"{item}." if str(item)[-1]!='.' else item for item in correct_answer_list]
    # response_list = [ f"{item}." if str(item)[-1]!='.' else item for item in response_list]
    # gptJ = GPTJudge(args)

    # scores = gptJ.score(init_question_list, correct_answer_list, response_list)

    # df[f'{args.key1}_scores'] = scores

    # df['question'] = init_question_list

    # # for s, q, ca, ga in zip(scores, list(df['question']), list(df[args.key2]), list(df[args.key1])):
    # #     print(f"[QUESTION]: {q}\n[CORRECT ANS]: {ca}\n[GEN ANS]: {ga}\t[SCORE]: {s}")

    # print(f"[OUTPUT]: final avg score: {np.average(scores)}")
    # print(f"[OUTPUT]: final helpful answer ratio: {np.sum(np.array(scores)>3)/len(scores)}")
    # if args.output_data_dir != "":
    #     # 将DataFrame写入Excel文件
    #     path = os.path.join(args.output_data_dir, f"result_{time_string}.xlsx")
    #     df.to_excel(path, index=False)
    #     print(f"save result into {path}")
    


if __name__ == "__main__":

    main()