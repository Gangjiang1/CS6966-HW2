import argparse
import numpy as np
import torch
import jsonlines 
import random 
import os
# 创建目录 'out/'，如果它不存在
os.makedirs('out/', exist_ok=True)
os.makedirs('train_out/', exist_ok=True)

# 加载args.task_name 任务的 args.subtask 数据集
def newyorker_caption_contest_data(args):
    from datasets import load_dataset
    dset = load_dataset(args.task_name, args.subtask)

    res = {}
    for spl, spl_name in zip([dset['train'], dset['validation'], dset['test']],
                            ['train', 'val', 'test']):
        cur_spl = []
        for inst in list(spl):
            inp = inst['from_description']
            targ = inst['label']
            cur_spl.append({'input': inp, 'target': targ, 'instance_id': inst['instance_id'], 'image': inst['image'], 'caption_choices': inst['caption_choices']})
        
            #'input' is an image annotation we will use for a llama2 e.g. "scene: the living room description: A man and a woman are sitting on a couch. They are surrounded by numerous monkeys. uncanny: Monkeys are found in jungles or zoos, not in houses. entities: Monkey, Amazon_rainforest, Amazon_(company)."
            #'target': a human-written explanation 
            #'image': a PIL Image object
            #'caption_choices': is human-written explanation

        res[spl_name] = cur_spl
    return res
# idefics该模型接受任意图像和文本输入序列并生成输出文本
def newyorker_caption_contest_idefics(args): 
    from transformers import IdeficsForVisionText2Text, AutoProcessor

    print("Loading model")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = IdeficsForVisionText2Text.from_pretrained(args.idefics_checkpoint, torch_dtype=torch.bfloat16).to(device)
    processor = AutoProcessor.from_pretrained(args.idefics_checkpoint)

    print("Loading data")
    nyc_data = newyorker_caption_contest_data(args)
    random.seed(args.seed)
    nyc_data_five_val = random.sample(nyc_data['val'],5)
    nyc_data_train_two = random.sample(nyc_data['train'],2)

    prompts = []
    for train_inst in nyc_data_train_two:
        train_inst['image'].save(f"train_out/{train_inst['instance_id']}.jpg")
    for val_inst in nyc_data_five_val:
        # ======================> ADD YOUR CODE TO DEFINE A PROMPT WITH TWO TRAIN EXAMPLES/DEMONSTRATIONS/SHOTS <======================
        # Each instace has a key 'image' that contains the PIL Image. You will give that to the model as input to "show" it the image instead of an url to the image jpg file.
        print('Val instance target:')
        print(val_inst['target'])
        prompts.append(["User: Describe the content of this cartoon and explain why this joke relates to the scenario with an explanation",
                        f"The caption is: {nyc_data_train_two[0]['caption_choices']}, the cartoon content is: {nyc_data_train_two[0]['input']}",
                        nyc_data_train_two[0]['image'],
                        "<end_of_utterance>",
                        f"\nAssistant: {nyc_data_train_two[0]['target']}",
                        "\nUser:",
                        f"The caption is: {nyc_data_train_two[1]['caption_choices']}, the cartoon content is: {nyc_data_train_two[1]['input']}",
                        nyc_data_train_two[1]['image'],
                        "And explain this one.<end_of_utterance>",
                        f"\nAssistant: {nyc_data_train_two[1]['target']}",

                        "\nUser: Explain the next cartoon and give an explanation:",
                        f"The caption is: {val_inst['caption_choices']}. The content is: {val_inst['input']}",
                        val_inst['image'],
                        "<end_of_utterance>",
                        
                        "\nAssistant:",])
        
        # I'm saving images to `out`` to be able to see them in the output folder
        val_inst['image'].save(f"out/{val_inst['instance_id']}.jpg")
    print('Prompts:\n',prompts)
    # --batched mode
    inputs = processor(prompts, add_end_of_utterance_token=False, return_tensors="pt").to(device)
    # --single sample mode
    #inputs = processor(prompts[0], return_tensors="pt").to(device)
    # Generation args
    exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
    # 可以使用 bad_words_ids 来指定要在生成的文本中避免的单词或标记，如 "<image>" 和 "<fake_token_around_image>"
    # model.generate 生成
    generated_ids = model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, max_length=1024)
    # 将模型的生成转化为文本
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    print('Generated text:\n', generated_text)
    print('One by one generation:')
    for i, t in enumerate(generated_text):
        print(f"{i}:\n{t}\n")

        # 
        gen_expl = t.split("Assistant:")[-1]
        nyc_data_five_val[i]['generated_idefics']=gen_expl

    # ======================> You will need to `mkdir out`
    filename = 'out/val.jsonl'
    with jsonlines.open(filename, mode='w') as writer:
        for item in nyc_data_five_val:
            del item['image']
            writer.write(item)

    filename = 'out/train.jsonl'
    with jsonlines.open(filename, mode='w') as writer:
        for item in nyc_data_train_two:
            del item['image']
            writer.write(item)
        

def newyorker_caption_contest_llama2(args): 
    print ("Loading data")
    nyc_data_five_val = []
    with jsonlines.open('out/val.jsonl') as reader:
        for obj in reader:
            nyc_data_five_val.append(obj)

    nyc_data_train_two = []
    with jsonlines.open('out/train.jsonl') as reader:
        for obj in reader:
            nyc_data_train_two.append(obj)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    print(nyc_data_train_two)
    print("................")
    print(nyc_data_five_val)
    print("Loading model")
    '''
    Ideally, we'd do something similar to what we have been doing before: 

        tokenizer = AutoTokenizer.from_pretrained(args.llama2_checkpoint, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(args.llama2_checkpoint, torch_dtype=torch.float16, device_map="auto")
        tokenizer.pad_token = tokenizer.unk_token_id
        
        prompts = [ "our prompt" for val_inst in nyc_data_five_val]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        output_sequences = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        generated_text = [tokenizer.decode(s, skip_special_tokens=True) for s in output_sequences]

    But I cannot produce text with this prototypical code with HF llama2. 
    Thus we will use pipeline instead. 
    '''
    import transformers
    from transformers import AutoTokenizer
    # llamaid = "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"

    tokenizer = AutoTokenizer.from_pretrained(args.llama2_checkpoint)
    pipeline = transformers.pipeline(
        "text-generation",
        model=args.llama2_checkpoint,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    for i, val_inst in enumerate(nyc_data_five_val):         
        # ======================> ADD YOUR CODE TO DEFINE A PROMPT WITH TWO TRAIN EXAMPLES/DEMONSTRATIONS/SHOTS <======================
        prompt = f"Describe the content of this cartoon and give an explanation. {nyc_data_train_two[0]['input']} [/INST] {nyc_data_train_two[0]['target']} </s><s>[INST]\
        Describe the content of this cartoon and give an explanation. {nyc_data_train_two[1]['input']} [/INST] {nyc_data_train_two[1]['target']} </s><s>[INST]\
        Please describe the next cartoon and give an explanation. {val_inst['input']} [/INST]"

        sequences = pipeline(
            prompt,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            max_length=1024,
        )
        
        gen_expl = sequences[0]['generated_text'].split("/INST] ")[-1]
        nyc_data_five_val[i]['generated_llama2']=gen_expl

    filename = 'out/val.jsonl'
    with jsonlines.open(filename, mode='w') as writer:
        for item in nyc_data_five_val:
            writer.write(item)
    print("Generated and saved")

if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 获取当前工作目录并设置为默认输出路径
    current_directory = os.getcwd()
    parser.add_argument('--seed', default="1421910",type=int, help='Random seed set to your uNID') # <======================> 
    parser.add_argument('--output_dir', default=current_directory, type=str, help='Directory where model checkpoints will be saved')
    parser.add_argument('--task_name', default="jmhessel/newyorker_caption_contest",  type=str, help='Name of the task that will be used by huggingface load dataset')    
    parser.add_argument('--subtask', default="explanation", type=str, help="The contest has three subtasks: matching, ranking, explanation")
    parser.add_argument('--idefics_checkpoint', default="HuggingFaceM4/idefics-9b-instruct", type=str, help="The hf name of an idefics checkpoint")
    parser.add_argument('--llama2_checkpoint', default="daryl149/llama-2-7b-chat-hf", type=str, help="The hf name of a llama2 checkpoint")
    # 解析命令行参数 
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    newyorker_caption_contest_idefics(args)
    newyorker_caption_contest_llama2(args)
