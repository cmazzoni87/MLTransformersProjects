import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
PATH_DOC = os.path.join(os.path.dirname(__file__), 'Documents')
PATH_OUT = os.path.join(os.path.dirname(__file__), 'Output')

model = T5ForConditionalGeneration.from_pretrained('t5-large')
tokenizer = T5Tokenizer.from_pretrained('t5-large')
device = torch.device('cuda')
chap_n = 0
with open(os.path.join(PATH_DOC, 'literature_and_revolution_byTrosky.txt'), 'r') as marx:
    with open(os.path.join(PATH_OUT, 'literature_and_revolution_byTrosky_Summarized.txt'), 'w') as out:
        literature = marx.read()
        for chapter in literature.split('Chapter '):
            preprocess_text = chapter.strip().replace("\n", " ")
            tokens = [' '.join(preprocess_text.split()[i:i + 340]) for i in range(0, len(preprocess_text.split()), 350)]
            head = "\n\nChapter {} Summary: \n".format(chap_n)
            print(head)
            out.write(head)
            torch.cuda.empty_cache()
            for token in tokens:
                t5_prepared_Text = "summarize: " + token
                tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
                summary_ids = model.to(device).generate(tokenized_text,
                                             num_beams=4,
                                             no_repeat_ngram_size=2,
                                             min_length=30,
                                             max_length=60,
                                             early_stopping=True)
                output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                out.write('{}\n'.format(output))
            chap_n += 1
        out.close()

