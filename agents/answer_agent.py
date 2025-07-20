#!/usr/bin/python3

import re
import json

from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict, Any
from .answer_model2 import AAgent as AAgent2
from .answer_model import AAgent

class AnsweringAgent(object):
    r"""Agent responsible for answering MCQ questions with confidence scoring"""
    
    def __init__(self, select_prompt1: bool = True,router: bool = True, problem:str = "family",**kwargs):
        self.agent = AAgent2(**kwargs)
        self.select_prompt1 = select_prompt1
        self.router = router
        self.problem = problem
    
    def build_prompt(self, question_data: Dict[str, str|Any]) -> Tuple[str, str]:
        """Generate an answer to the given MCQ question with confidence and reasoning"""

        bloodline_prompt = """
            If you are asked to find the relationship between any two entities, then use the following instructions:
                1. Create a graph out of the given information
                2. Now, begin track the one of the two nodes across whom the relationship is to be determined
                3. Treat this as a depth first algorithm and keep doing this until you find the other node mentioned in the question.
                4. Finally, by carefully understanding the path properties between the two nodes, you will realize their relationship.

                **PLEASE NOTE THAT IF YOU ARE UNABLE TO FIND ANY LINKS BETWEEN THE NODES, GO FOR THE COUSINS OPTION***

                For example,

                {
                "topic": "Puzzles involving generations and family tree logic",
            "question": "A is the father of C and D is the son of B. E is the brother of A if C is the sister of D, how is B related to E?",
            "choices": [
                "A) Brother",
                "B) Brother-in-law",
                "C) Sister",
                "D) Sister-in-law"
            ],
            "explanation": "C and D are brother and sister that measn A and B are husband and wife and wife will be sister in law of husband’s brother. So the correct option is 'C'",
            "answer": "C"
        },
        {
            "topic": "Puzzles involving generations and family tree logic",
            "question": "A and B are brothers. C and D are sisters. A’s son is D’s brother. How is C related to B?",
            "choices": [
                "A) Uncle",
                "B) Niece",
                "C) Father",
                "D) Mother"
            ],
            "explanation": "C D and A’s son are brother and sisters, so brother’s daughter will be B’s niece. Hence, option 'B'.",
            "answer": "B"
        },
        {
            "topic": "Puzzles involving generations and family tree logic",
            "question": "Rohan said to Rashmi, ‘ your mother’s husband’s sister is my aunt.’ How is Rashmi related to Rohan?",
            "choices": [
                "A) Daughter",
                "B) Sister",
                "C) Mother",
                "D) Granddaughter"
            ],
            "explanation": "mother’s husband is father and father’s sister will be Rohan’s aunt so that means Rohan and Rashmi are brother and sister. Hence, option 'B'.",
            "answer": "B"
        }
        """

        truth_teller = """
            If you are asked any question regarding truth and lies, about truth-telling and liar entities, then use the following guide to answer:
                1. Begin by assuming any one node is either truth telling or lying.
                2. With this assumption, assign values to every node
                3. If, at any point, the value assumed by the assumption leads to contradictions, then we conclude that the assumed value of the node is incorrect.
                4. So now we know that the node is what exactly. Keep doing with other actors this until you have deduced the state (lying or turth telling) of all nodes.
                5. Now, please proceed with the answer now.

                For example,
                {
                "topic": "Truth-teller and Liar Problems",
            "question": "On an island, three inhabitants A, B, and C make statements: A says, 'B is a liar.' B says, 'C is a liar.' C says, 'A and B are of different types.' If exactly one is liar, who are the truth-tellers?",
            "choices": [
                "A) A and B",
                "B) B and C",
                "C) A and C",
                "D) None"
            ],
            "explanation": "If we assume A to be the liar, then it means it lied about B being a liar. So that makes B a truth-teller. So it means that according to B's statement, C is a liar. So A and B are of the same type. But A is a liar and B is a truth-teller. This leads to a contradiction. Hence, A has to be a truth-teller. So this makes B a liar. This indicates that C is a truth-teller and A and B are of different types. So A and C are the truth tellers. This makes the third option, C, the correct option.",
            "answer": "C"
        },
        {
        "topic": "Truth-teller and Liar Problems",
    "question": "Two people, Red and Blue, stand before you. Red says, “We are both knaves.” What are they really?",
    "choices": [
      "A) Red is truth-teller",
      "B) Blue is is truth-teller",
      "C) Both are truth-tellers",
      "D) Both are liars"
    ],
    "explanation": "Red cannot be a truth-teller, because then he would be lying by saying he is a liar. Therefore he is a liar, and his statement is a lie, meaning that Blue must be a truth-teller. Hence, option 'B'.",
    "answer": "B"
    },
    {
    "topic": "Truth-teller and Liar Problems",
    "question": "A new island is discovered, with a third type of inhabitant: spies, who can either tell the truth or lie. You encounter three people, one of each, but you don’t know which is which. Red says, “I am a truth-teller.” Blue says, “Red is telling the truth.” Green says, “I am the spy.” Who is what?",
    "choices": [
      "A) Red is truth-teller",
      "B) Blue is is truth-teller",
      "C) Both are truth-teller",
      "D) Green is the truth-teller"
    ],
    "explanation": "Could Blue be lying? That would mean that Red is also lying, so Green would have to be the truth-teller. But Green cannot be the truth-teller, because the truth-teller would not lie about who he is. Therefore Blue is telling the truth, so Red is the truth-teller. Since Blue is a non-truth-teller who told the truth, he must be the spy, making Green the liar. Hence, option 'B'.",
    "answer": "B"
}
        """
        seating_problem = """
            If you asked a question regarding the position/location of an entity with or without respect to some other entity, use the following instructions:
                1. Follow the question and create a graphical representation of the arrangement.
                2. Note that the graph will always be either a linear graph (a line) or a circle.
                3. Once you have the graph, look for the entity that the question is looking for. Find there is another entity involved, take that into account and arrive on the final anaswer.
                For example, 
                {
                    "topic": "Seating Arrangements (Linear, Circular)",
                    "question": "Five people A, B, C, D and E are seated about a round table. Every chair is spaced equidistant from adjacent chairs. i. C is seated next to A. ii. A is seated two seats from D. iii. B is not seated next to A. On the basis of above information, which of the following must be true? 1. D is seated next to B. 2. E is seated next to A. 3. D and C are separated by two seats. Select the correct answer using the code given below:",
                    "choices": [
                      "A) 1 only",
                      "B) 1 and 2 only",
                      "C) 3 only",
                      "D) Neither 1 nor 2 nor 3"
                    ],
                    "explanation": "Given that, Five people A, B, C, D and E are, seated about a round table. Every chair is spaced equidistant from adjacent chairs. (i) C is seated next to A. (ii) A is seated two seats from D. (iii) B is not seated next to A. Now, It can be concluded from the given information Two cases : E A C D B or E A C B D 1. D is seated next to B. A is not seated next to B and A is two seats from D, thus D must seat next to B. EACBD or EACDB. Hence statement 1 is correct 2. E is seated next to A. EACBD or EACDB. Hence statement 2 is correct. 3. D and C are separated by two seats. EACBD or EACDB. Hence statement 3 is incorrect. Hence, option 'B'.",
                    "answer": "B"
}
{
                "topic": "Seating Arrangements (Linear, Circular)",
            "question": "Five friends from different countries sit circularly. The Italian sits opposite the tea-drinker. The Japanese sits two seats to the left of the coffee-drinker. The Brazilian drinks juice. The Chinese is adjacent to the American. Milk is drunk by someone adjacent to juice. Who drinks coffee?",
            "choices": [
                "A) American",
                "B) Chinese",
                "C) Japanese",
                "D) Brazilian"
            ],
            "explanation": "Brazilian (juice) has milk adjacent. Italian opposite tea. Japanese -> 2 seats left of coffee -> American must be coffee (Chinese adjacent to American, not conflicting with other constraints). Hence, option 'A'.",
            "answer": "A"
        },
        {
            "topic": "Seating Arrangements (Linear, Circular)",
            "question": "Eight people A, B, C, D, E, F, G, H sit around a circular table. Four face the center, and four face outward. A is third to the left of B, who faces the opposite direction of D. C (a doctor) sits adjacent to both E and F. G faces the center and is two seats to the left of H, who is not adjacent to B. If E faces outward, who is the engineer?",
            "choices": [
                "A) G",
                "B) H",
                "C) F",
                "D) D"
            ],
            "explanation": "H's position and facing direction (outward) are derived from clues. Since C is a doctor and professions aren't repeated, H must be the engineer.Hence, option 'B'.",
            "answer": "B"
        },
        {
            "topic": "Seating Arrangements (Linear, Circular)",
            "question": "Twelve people sit in two parallel rows of six each. Front row faces south, back row faces north. P is behind Q, who is third from the left end. R faces T, who is adjacent to S. U is two places to the right of V in the same row. If W is at an extreme end in the back row, who sits at the front row's extreme right?",
            "choices": [
                "A) S",
                "B) T",
                "C) U",
                "D) V"
            ],
            "explanation": "Q is third from left in front row; P is behind Q. R faces T (adjacent to S). U and V positions fix S at the front right.Hence, option 'A'.",
            "answer": "A"
        }
        """
        
        sys_prompt1 = "You are an expert in quantitative aptitude for competitive exams, solving MCQs with step-by-step reasoning before selecting the correct answer."
        sys_prompt2 = (
            "You are an expert answer agent specializing in solving multiple-choice questions (MCQs) that test "
            "quantitative aptitude skills, as seen in top-tier competitive exams. "
            "You have a deep understanding of logical reasoning, puzzles, and analytical problem-solving under exam conditions. "
            "For each question, think step by step using a clear chain-of-thought approach. "
            "Break down the problem, analyze all options, eliminate distractors, and then confidently select the correct answer. "
            "Always explain your reasoning before finalizing your choice."
            
        )

        if self.router:
            tmpl = (
                'For the following questions answer the what type of problem it is\n'
                "eg. Question: In a family gathering, Aryan pointed to a man and said, 'He is the son of my mother's only sister, who is married to the only brother of my paternal grandfather's daughter-in-law.' How is the man related to Aryan? answer: family_tree"
                "eg. Question: Five friends from different countries sit circularly. The Italian sits opposite the tea-drinker. The Japanese sits two seats to the left of the coffee-drinker. The Brazilian drinks juice. The Chinese is adjacent to the American. Milk is drunk by someone adjacent to juice. Who drinks coffee? answer: seating_problem"
                "eg. Question: On an island, three inhabitants A, B, and C make statements: A says, 'B is a liar.' B says, 'C is a liar.' C says, 'A and B are of different types.' If exactly two are liars, who is the truth-teller? answer: truth_question"
                'Question: {}\n'
                'RESPONSE FORMAT: Strictly generate a valid JSON object as shown below:\n'
                '{{\n'
                '    "reasoning": "Brief explanation within 100 words"\n'
                '    "answer": "one of the following words :- family_tree, seating_problem, truth_question",\n'
                '}}'
            )
            
            prompt = tmpl.format(
                question_data['question']
                # self._format_choices(question_data['choices'])
            )
            
        else:
            tmpl = (
                'INSTRUCTIONS FOR ANSWERING:\n'
                '1. Carefully read and understand what is being asked might be correct or incorrect.\n'
                '2. Consider why each choice is correct.\n'
                '3. There is only **ONE OPTION** correct.\n'
                '4. Provide reasoning within 100 words\n\n'
                'RESPONSE FORMAT: Strictly generate a valid JSON object as shown below:\n'
                '{{\n'
                '    "reasoning": "Brief explanation within 100 words"\n'
                '    "answer": "One letter string in of one of the following letter:- "A", "B", "C", "D"",\n'
                '}}'
                '{}'
                'Question: {}\n'
                'Choices: {}\n\n'
                
                'RESPONSE FORMAT: Strictly generate a valid JSON object as shown below:\n'
                '{{\n'
                '    "reasoning": "Brief explanation within 100 words"\n'
                '    "answer": "One letter string in of one of the following letter:- "A", "B", "C", "D"",\n'
                '}}'
            )

            if self.problem == "family":
                prompt = tmpl.format(
                    bloodline_prompt,
                    question_data['question'],
                    self._format_choices(question_data['choices'])
                )
            elif self.problem == "truth":
                prompt = tmpl.format(
                    truth_teller,
                    question_data['question'],
                    self._format_choices(question_data['choices'])
                )
            else:
                prompt = tmpl.format(
                    seating_problem,
                    question_data['question'],
                    self._format_choices(question_data['choices'])
                )
            
        return prompt, sys_prompt1 if self.select_prompt1 else sys_prompt2
    
    def answer_question(self, question_data: Dict|List[Dict], **kwargs) -> Tuple[List[Dict], int|None, float|None]:
        """Generate answer(s) for the given question(s)"""
        if isinstance(question_data, list):
            prompt = []
            for qd in question_data:
                p, sp = self.build_prompt(qd)
                prompt.append(p)
        else:
            prompt, sp = self.build_prompt(question_data)
        
        resp, tl, gt = self.agent.generate_response(prompt, sp, **kwargs)

        if (isinstance(resp, list) and all(isinstance(r, str) for r in resp)) or isinstance(resp, str):
            return resp, tl, gt
        else:
            return '', tl, gt if not isinstance(resp, list) else [''] * len(resp), tl, gt
    
    def answer_batches(self, questions: List[Dict], batch_size: int = 5, **kwargs) -> Tuple[List[Dict], List[int | None], List[float | None]]:
        """Answer questions in batches"""
        answers = []
        tls, gts = [], []
        total_batches = (len(questions) + batch_size - 1) // batch_size
        pbar = tqdm(total=total_batches, desc="STEPS: ", unit="batch")
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i + batch_size]
            batch_answers, tl, gt = self.answer_question(batch_questions, **kwargs)
            answers.extend(batch_answers)
            tls.append(tl); gts.append(gt)
            pbar.update(1)
        
        # Handle last batch with less than batch_size
        # if len(questions) % batch_size != 0:
        #     batch_questions = questions[-(len(questions) % batch_size):]
        #     batch_answers = self.answer_question(batch_questions, **kwargs)
        #     answers.extend(batch_answers[0]); tls.append(batch_answers[1]); gts.append(batch_answers[2])
        #     pbar.update(1)
        pbar.close()
        return answers, tls, gts
    
    def count_tokens_a(self, text: str) -> int:
        """Count the number of tokens in the text using the agent's tokenizer"""
        if not hasattr(self.agent, 'tokenizer'):
            raise AttributeError("The agent does not have a tokenizer attribute.")
        return len(self.agent.tokenizer.encode(text, add_special_tokens=False))

    def filter_answers(self, ans: List[str|Dict[str, str]]) -> List[Dict[str, str]]:
        r"""Filter answers to ensure they are in the correct format"""
        def basic_checks(a1: Dict[str, str])->bool:
            # check required keys
            required_keys = ['answer']
            if all((key in a1) and isinstance(a1[key], str) for key in required_keys):
                if len(a1['answer']) == 1 and (a1['answer'] not in 'ABCDabcd'):
                    return False
                check_len = self.count_tokens_a(a1['answer'])
                if check_len < 50:
                    check_len += self.count_tokens_a(a1.get('reasoning', 'None'))
                    if check_len < 512:
                        # check answer format - EXTRA checks
                        # if len(a1['answer']) == 1 and a1['answer'].upper() in 'ABCD':
                        return True
            return False
    
        filtered_answers = []
        for i, a in enumerate(ans):
            if isinstance(a, dict):
                if basic_checks(a):
                    filtered_answers.append(a)
                else:
                    filtered_answers.append(None)
                    print(f"Skipping invalid answer at index {i}: {a}")
            elif isinstance(a, str):
                # Basic checks: at least with correct JSON format
                try:
                    a1 = json.loads(a)
                    if basic_checks(a1):
                        filtered_answers.append(a1)
                    else:
                        filtered_answers.append(None)
                        print(f"Skipping invalid answer at index {i}: {a}")
                except json.JSONDecodeError:
                    # If JSON decoding fails, skip this answer
                    print(f"Skipping invalid JSON at index {i}: {a}")
                    filtered_answers.append(None)
                    continue
            else:
                # If the answer is neither a dict nor a str, skip it
                print(f"Skipping unsupported type at index {i}: {type(a)}")
                filtered_answers.append(None)
        return filtered_answers

    def save_answers(self, answers: List[str], file_path: str|Path) -> None:
        """Save generated answers to a JSON file"""
        # check for existence of dir
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump([a for a in answers], f, indent=4)
    
    def _format_choices(self, choices: List[str]) -> str:
        r"""Format the choices for better readability"""
        formatted = []
        for choice in choices:
            # Ensure each choice starts with a letter if not already formatted
            if not re.match(r'^[A-D]\)', choice.strip()):
                # Extract letter from existing format or assign based on position
                letter = chr(65 + len(formatted))  # A, B, C, D
                formatted.append(f"{letter}) {choice.strip()}")
            else:
                formatted.append(choice.strip())
        return " ".join(formatted)


# Example usage
if __name__ == "__main__":
    import json
    import yaml
    import argparse
    from utils.build_prompt import auto_json, option_extractor_prompt
    import time

    # Record the start time
    start_time = time.time()
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # python -m agents.answer_agent --input_file outputs/filtered_questions.json --output_file outputs/answers.json --batch_size 5 --verbose
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    argparser = argparse.ArgumentParser(description="Run the Answering Agent")
    argparser.add_argument("--input_file", type=str, default="outputs/filtered_questions.json", help="Path to the input JSON file with questions")
    argparser.add_argument("--output_file", type=str, default="outputs/answers.json", help="Path to save the answers")
    argparser.add_argument("--batch_size", type=int, default=10, help="Batch size for processing questions")
    argparser.add_argument("--verbose", action='store_true', help="Enable verbose output")
    args = argparser.parse_args()

    SELECT_PROMPT1 = False  # Use the first system prompt for answering
    
    # Load sample questions (assuming they're saved from QuestioningAgent)
    with open(args.input_file, 'r') as f:
        sample_questions = json.load(f)
    
    agent = AnsweringAgent(select_prompt1=SELECT_PROMPT1, router =True)
    
    # gen_kwargs = {"tgps_show": True, "max_new_tokens": 512, "temperature": 0.1, "top_p": 0.9, "do_sample": True}
    gen_kwargs = {"tgps_show": True}
    with open("agen.yaml", "r") as f: gen_kwargs.update(yaml.safe_load(f))
    answer, tls, gts = agent.answer_batches(
        questions=sample_questions,
        batch_size=args.batch_size,
        **gen_kwargs
    )
    ans = []
    quest_q = {
        'seating_problem': [],
        'family_tree': [],
        'truth_question': [],
    }
    for idx, (q, a) in enumerate(zip(sample_questions, answer)):
        try:
            a = json.loads(a)
            # print(a['answer'])
            if all(k in a for k in ['answer']):
                # ++++++++++++++++++++++++++
                # TODO: IMPROVE THE FOLLOWING
                if a["answer"]=="seating_problem":
                    quest_q[a["answer"]].append((idx, q))
                elif a["answer"]=="family_tree":
                    quest_q[a["answer"]].append((idx, q))
                else:
                    quest_q[a["answer"]].append((idx, q))
                    # agent.agent.generate_response(option_extractor_prompt(a['answer'], q['choices']))
                # ++++++++++++++++++++++++++
            else:
                quest_q["family_tree"].append((idx, q))
                # the dictionary is not as expected. So extract it using the same model: Self-Reflection
                #Do some thing here
                # prompt = (
                #     'Extract **ONLY** the answer and reasoning while discarding the rest.\n\n'
                    
                #     'String:\n'
                #     '{}\n\n'

                #     'Given Format:\n'
                #     '{{\n'
                #     '    "answer": "Only the option letter (A, B, C, or D)",\n'
                #     '    "reasoning": "..."\n'
                #     '}}'
                # )
                # a = agent.agent.generate_response(prompt.format(json.dumps(a, indent=4)))
        except json.JSONDecodeError:
            quest_q["family_tree"].append((idx, q))
            # a = agent.agent.generate_response(auto_json(a))
        # ans.append(a)
    # sft_kwargs = {'adapter_type':'sft'}
    # grpo_kwargs = {'adapter_type':'grpo'}
    agents = {
        'seating_problem': AnsweringAgent(select_prompt1=SELECT_PROMPT1, router = False, problem = "seating"),
        'family_tree': AnsweringAgent(select_prompt1=SELECT_PROMPT1, router = False, problem = "family"),
        'truth_question': AnsweringAgent(select_prompt1=SELECT_PROMPT1, router = False, problem = "truth"),
    }
    args.batch_size = 10
    for key in quest_q:
        # for (idx, q) in quest_q:
        answer, tls, gts = agents[key].answer_batches(
        questions=list(map(lambda x: x[1], quest_q[key])),
        batch_size=args.batch_size,
        **gen_kwargs
        )

        for ((idx, q), a) in zip(quest_q[key], answer):
            if args.verbose:
                print(f"\n=== Question {idx+1} ===")
                print(f"Question: {q.get('question', 'N/A')}")
                print(f"Expected: {q.get('answer', 'N/A')}")
                print(f"Model Answer:\n{a}")
            try:
                a = json.loads(a)
                # print(a['answer'])
                if all(k in a for k in ['answer', 'reasoning']):
                    # ++++++++++++++++++++++++++
                    # TODO: IMPROVE THE FOLLOWING
                    a['answer'] = a['answer'][:1]
                    if a['answer'] not in "abcdABCD":
                        a['answer'] = "B"
                        # a['answer'] = agent.agent.generate_response(option_extractor_prompt(a['answer'], q['choices']))
                    # ++++++++++++++++++++++++++
                else:
                    # the dictionary is not as expected. So extract it using the same model: Self-Reflection
                    prompt = (
                        'Extract **ONLY** the answer and reasoning while discarding the rest.\n\n'
                        
                        'String:\n'
                        '{}\n\n'
    
                        'Given Format:\n'
                        '{{\n'
                        '    "answer": "Only the option letter "A", "B", "C", or "D"",\n'
                        '    "reasoning": "..."\n'
                        '}}'
                    )
                    a = agent.agent.generate_response(prompt.format(json.dumps(a, indent=4)))
            except json.JSONDecodeError:
                a = agent.agent.generate_response(auto_json(a))
            ans.append((idx, a))
    ans.sort()
    ans = list(map(lambda x: x[1], ans))
    if args.verbose:
        if gen_kwargs.get('tgps_show', False):
            for idx, (tl, gt) in enumerate(zip(tls, gts)):
                print(f"BATCH - {idx}")
                print(f"Tokens: {tl}, Time: {gt:.3f} seconds")
                print(f"TGPS: {tl/gt:.3f} seconds")
            print("\n" + "="*50)
            print(f"Total Time: {sum(gts):.3f} seconds; Total Tokens: {sum(tls)}; TGPS: {sum(tls)/max(0.1, sum(gts)):.3f} seconds")
    
    # Save answers
    agent.save_answers(ans, args.output_file)
    filtered_file_name = args.output_file.replace("answers.json", "filtered_answers.json")
    agent.save_answers(agent.filter_answers(ans), filtered_file_name)
    print(time.time()-start_time)