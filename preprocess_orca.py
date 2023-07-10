from datasets import load_dataset

dataset = load_dataset('Open-Orca/OpenOrca', streaming=True)

system_prompts = []
count = 0
for idx, item in enumerate(dataset['train']):
    prompt = item['system_prompt'] + item['question']
    if prompt not in system_prompts:
        system_prompts.append(prompt)
        count += 1
        print(prompt)
    if count > 100: break

