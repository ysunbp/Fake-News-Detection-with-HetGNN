import json
import os

from tqdm import tqdm
from transformers import (DataCollatorForLanguageModeling,
                          LineByLineTextDataset, Trainer, TrainingArguments,
                          XLMRobertaForMaskedLM, XLMRobertaTokenizer)


def convert_weibo_text_into_line_by_line(weibo_dir, line_by_line_f, mini_size=float('inf')):
    text_list = []
    for fname in tqdm(os.listdir(weibo_dir), desc='reading weibo text'):
        with open(os.path.join(weibo_dir, fname), 'r') as fin:
            posts = json.load(fin)
        text_list.append(posts[0]["text"])
        # text_list.extend([post["user_description"]
                        #   for post in posts])  # 136 MB
        if len(text_list) > mini_size:
            break
    text_list = [t for t in text_list if len(t) > 0]
    with open(line_by_line_f, 'w') as fout:
        fout.write('\n'.join(text_list) + '\n')


if __name__ == '__main__':

    line_by_line_f = '/rwproject/kdd-db/20-rayw1/data/line_by_line_post.txt'

    model_in = 'xlm-roberta-base'
    config_tag = '-post'
    model_out = '/rwproject/kdd-db/20-rayw1/language_models/' + model_in + config_tag
    output_dir = '/rwproject/kdd-db/20-rayw1/language_models/output' + config_tag

    convert_weibo_text_into_line_by_line(
        weibo_dir='/rwproject/kdd-db/20-rayw1/rumdect/weibo_json', line_by_line_f=line_by_line_f)

    print('Loading models...')
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    model = XLMRobertaForMaskedLM.from_pretrained(model_in, return_dict=True)

    print('Loading dataset...')
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=line_by_line_f,
        block_size=128,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,  # 8 => CUDA out of memory on raymond's server
        save_steps=8000,  # 8000 ~ save every 15 min
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    print('Training...')
    trainer.train()
    trainer.save_model(model_out)
    