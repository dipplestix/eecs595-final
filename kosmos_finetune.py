#!/usr/bin/env python
# coding: utf-8

# In[56]:

import sys

import torch
import json
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoTokenizer
import torch
from PIL import Image
import pickle
import random
import peft
from trl import SFTTrainer
from torch.utils.data import Dataset, DataLoader
import copy
import wandb
from nltk.tokenize import word_tokenize
from tqdm import tqdm


# In[18]:


wandb.login()


# In[19]:


wandb.init(
    project="KOSMOS-2-finetune",
    config={
        "r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "modules_to_save": 'kosmos-2',

    },
)


# In[ ]:


model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")


# In[ ]:


tokenizer = AutoTokenizer.from_pretrained("microsoft/kosmos-2-patch14-224")
tokenizer.pad_token = tokenizer.eos_token


# In[ ]:


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device, file=sys.stderr)
model.to(device)


# In[29]:


print(model, file=sys.stderr)


# In[ ]:


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    , file=sys.stderr)


# In[15]:


# model_clone = copy.deepcopy(model)
print_trainable_parameters(model)


# In[39]:


from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["kosmos2"],
    target_modules=["k_proj", "v_proj", "q_proj", "out_proj", "fc1", "fc2", "lm_head"]
)
peft_model = get_peft_model(model, lora_config)


# In[40]:


print_trainable_parameters(peft_model)


# In[41]:


for param in model.parameters():
    param.requires_grad = False


# In[43]:


# for param in model.qformer.parameters():
#     param.requires_grad = True


# In[44]:


def get_data(qid, data):
    image = data[qid]['imageId']
    question = data[qid]['question']
    answer = data[qid]['answer']
    full_answer = data[qid]['fullAnswer']
    return image, question, answer, full_answer


# In[45]:


f = open('/scratch/chaijy_root/chaijy0/josuetf/mac-network-v2/data/train_balanced_questions.json')
data = json.load(f)


# In[46]:


random_keys = random.sample(list(data.keys()), 20000)
train_data = {key: data[key] for key in random_keys}


# In[47]:


with open("train_questions_finetune.json", 'w') as f:
    json.dump(train_data, f)


# In[48]:


# Moslty from: https://github.com/huggingface/notebooks/blob/main/peft/Fine_tune_BLIP2_on_an_image_captioning_dataset_PEFT.ipynb
class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=item["image"], text=item["question"], 
                                  padding="max_length", max_length=60, return_tensors="pt")
        # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        # encoding["question"] = item["question"]
        encoding["answer"] = item["answer"]
        return encoding

def collate_fn(batch):
    # pad the input_ids and attention_mask
    processed_batch = {}
    for key in batch[0].keys():
        if key  not in ["answer"]:
            processed_batch[key] = torch.stack([example[key] for example in batch])
        else:
            text_inputs = processor.tokenizer(
                [example["answer"] for example in batch], padding=True, return_tensors="pt"
            )
            processed_batch["input_ids"] = text_inputs["input_ids"]
            processed_batch["attention_mask"] = text_inputs["attention_mask"]
    return processed_batch



# In[49]:


def run_example(prompt, image):
    
    image = Image.open(image)

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs.to(device)
    generated_ids = model.generate(
      pixel_values=inputs["pixel_values"],
      input_ids=inputs["input_ids"],
      attention_mask=inputs["attention_mask"],
      image_embeds=None,
      image_embeds_position_mask=inputs["image_embeds_position_mask"],
      use_cache=True,
      max_new_tokens=128,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    _processed_text = processor.post_process_generation(generated_text, cleanup_and_extract=False)
    processed_text, entities = processor.post_process_generation(generated_text)
    
    return processed_text, entities, _processed_text


# In[53]:


def PromptPipeline(prompt, image):
    processed_text, entities, _processed_text = run_example(prompt, image)
    # Draw the bounding bboxes
    # bounded_image = draw_entity_boxes_on_image(image, entities, show=True)
    return processed_text, entities, _processed_text


# In[57]:


# dataset = []
# for k in tqdm(train_data):
#     image, question, answer, _ = get_data(k, train_data)
#     question_formatted = f'{question} Answer simply with one word:'
#     PILimage = Image.open('/scratch/chaijy_root/chaijy0/josuetf/mac-network/data/images/'+image+'.jpg') 
#     answer = word_tokenize(PromptPipeline(question_formatted, f"/scratch/chaijy_root/chaijy0/josuetf/mac-network/data/images/{image}.jpg")[0].split(":")[1])[0]
#     dataset.append({'image': PILimage, 'question': question_formatted, 'answer': answer})

# with open("finetune_dataset.pkl", 'wb') as file:
#     pickle.dump(dataset, file, protocol=pickle.HIGHEST_PROTOCOL)

with open("finetune_dataset.pkl", "rb") as file:
    dataset = pickle.load(file)


# Create an instance of ImageCaptioningDataset
train_dataset = ImageCaptioningDataset(dataset, processor)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=100, collate_fn=collate_fn)


# In[37]:


print(peft_model.device, file=sys.stderr)


# In[44]:


optimizer = torch.optim.SGD(peft_model.parameters(), lr=5e-4)

device = "cuda" if torch.cuda.is_available() else "cpu"

peft_model.train()
for epoch in range(5):
    print(f'Epoch: {epoch}', file=sys.stderr)
    n = 0
    for idx, batch in enumerate(train_dataloader):        
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device, torch.float32)
        outputs = peft_model(input_ids=input_ids,
                        pixel_values=pixel_values,
                        labels=input_ids)

        loss = outputs.loss
        wandb.log({'training_loss': loss.item()})
        
        if (n+1) % 10 == 0:
            print(f'{n+1} Loss: {loss.item()}', file=sys.stderr)
        n += 1
        loss.backward()
    
        torch.nn.utils.clip_grad_norm_(peft_model.parameters(), 1.0)

        optimizer.step()
        optimizer.zero_grad()



# In[45]:


torch.save(peft_model.state_dict(), 'finetunekosmos15epochstatedict.torch')


# In[46]:


torch.save(peft_model, 'finetunekosmos15epoch.torch')


# In[47]:


wandb.finish()


# In[ ]:




