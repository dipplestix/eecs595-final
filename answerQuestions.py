import sys

print('starting 1', file=sys.stderr)

from nltk.tokenize import word_tokenize
import json
import pickle
import os
import cv2
import numpy as np
import os
import requests
import torch
import torchvision.transforms as T
from time import time

from PIL import Image
import requests

from transformers import AutoProcessor, AutoModelForVision2Seq

import torch

print('starting 2', file=sys.stderr)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device, file=sys.stderr)

print('loading', file=sys.stderr)

model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
model = model.to(device)
processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    
print('loaded', file=sys.stderr)

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


# def is_overlapping(rect1, rect2):
#     x1, y1, x2, y2 = rect1
#     x3, y3, x4, y4 = rect2
#     return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4)


# def draw_entity_boxes_on_image(image, entities, show=False, save_path=None):
#     """_summary_
#     Args:
#         image (_type_): image or image path
#         collect_entity_location (_type_): _description_
#     """
#     if isinstance(image, Image.Image):
#         image_h = image.height
#         image_w = image.width
#         image = np.array(image)[:, :, [2, 1, 0]]
#     elif isinstance(image, str):
#         if os.path.exists(image):
#             pil_img = Image.open(image).convert("RGB")
#             image = np.array(pil_img)[:, :, [2, 1, 0]]
#             image_h = pil_img.height
#             image_w = pil_img.width
#         else:
#             raise ValueError(f"invaild image path, {image}")
#     elif isinstance(image, torch.Tensor):
#         image_tensor = image.cpu()
#         reverse_norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
#         reverse_norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]
#         image_tensor = image_tensor * reverse_norm_std + reverse_norm_mean
#         pil_img = T.ToPILImage()(image_tensor)
#         image_h = pil_img.height
#         image_w = pil_img.width
#         image = np.array(pil_img)[:, :, [2, 1, 0]]
#     else:
#         raise ValueError(f"invaild image format, {type(image)} for {image}")

#     if len(entities) == 0:
#         return image

#     new_image = image.copy()
#     previous_bboxes = []
#     # size of text
#     text_size = 1
#     # thickness of text
#     text_line = 1  # int(max(1 * min(image_h, image_w) / 512, 1))
#     box_line = 3
#     (c_width, text_height), _ = cv2.getTextSize("F", cv2.FONT_HERSHEY_COMPLEX, text_size, text_line)
#     base_height = int(text_height * 0.675)
#     text_offset_original = text_height - base_height
#     text_spaces = 3

#     for entity_name, (start, end), bboxes in entities:
#         for (x1_norm, y1_norm, x2_norm, y2_norm) in bboxes:
#             orig_x1, orig_y1, orig_x2, orig_y2 = int(x1_norm * image_w), int(y1_norm * image_h), int(x2_norm * image_w), int(y2_norm * image_h)
#             # draw bbox
#             # random color
#             color = tuple(np.random.randint(0, 255, size=3).tolist())
#             new_image = cv2.rectangle(new_image, (orig_x1, orig_y1), (orig_x2, orig_y2), color, box_line)

#             l_o, r_o = box_line // 2 + box_line % 2, box_line // 2 + box_line % 2 + 1

#             x1 = orig_x1 - l_o
#             y1 = orig_y1 - l_o

#             if y1 < text_height + text_offset_original + 2 * text_spaces:
#                 y1 = orig_y1 + r_o + text_height + text_offset_original + 2 * text_spaces
#                 x1 = orig_x1 + r_o

#             # add text background
#             (text_width, text_height), _ = cv2.getTextSize(f"  {entity_name}", cv2.FONT_HERSHEY_COMPLEX, text_size, text_line)
#             text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2 = x1, y1 - (text_height + text_offset_original + 2 * text_spaces), x1 + text_width, y1

#             for prev_bbox in previous_bboxes:
#                 while is_overlapping((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2), prev_bbox):
#                     text_bg_y1 += (text_height + text_offset_original + 2 * text_spaces)
#                     text_bg_y2 += (text_height + text_offset_original + 2 * text_spaces)
#                     y1 += (text_height + text_offset_original + 2 * text_spaces)

#                     if text_bg_y2 >= image_h:
#                         text_bg_y1 = max(0, image_h - (text_height + text_offset_original + 2 * text_spaces))
#                         text_bg_y2 = image_h
#                         y1 = image_h
#                         break

#             alpha = 0.5
#             for i in range(text_bg_y1, text_bg_y2):
#                 for j in range(text_bg_x1, text_bg_x2):
#                     if i < image_h and j < image_w:
#                         if j < text_bg_x1 + 1.35 * c_width:
#                             # original color
#                             bg_color = color
#                         else:
#                             # white
#                             bg_color = [255, 255, 255]
#                         new_image[i, j] = (alpha * new_image[i, j] + (1 - alpha) * np.array(bg_color)).astype(np.uint8)

#             cv2.putText(
#                 new_image, f"  {entity_name}", (x1, y1 - text_offset_original - 1 * text_spaces), cv2.FONT_HERSHEY_COMPLEX, text_size, (0, 0, 0), text_line, cv2.LINE_AA
#             )
#             # previous_locations.append((x1, y1))
#             previous_bboxes.append((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2))

#     pil_image = Image.fromarray(new_image[:, :, [2, 1, 0]])
#     # if save_path:
#         # pil_image.save(save_path)
#     if show:
#         pil_image.show()

#     return new_image

def PromptPipeline(prompt, image):
    processed_text, entities, _processed_text = run_example(prompt, image)
    # Draw the bounding bboxes
    # bounded_image = draw_entity_boxes_on_image(image, entities, show=True)
    return processed_text, entities, _processed_text

f = open('data/val_balanced_questions.json')
data = json.load(f)

print("loaded val_balanced_questions", file=sys.stderr)

f = open('data/val_all_questions.json')
data_ref = json.load(f)

print("loaded val_all_questions.json", file=sys.stderr)

# This is from Andy's code
dev_val = {}
for k,v in data.items():
    dev_val[k]=v
    for qid in v['entailed']:
        dev_val[qid] = data_ref[qid]
    # n -= 1
    # if n == 0:
    #     break

with open("final_questions.json", 'w') as f:
    json.dump(dev_val, f)

f = open('final_questions.json')
new_data = json.load(f)

def get_data(qid, data):
    image = data[qid]['imageId']
    question = data[qid]['question']
    answer = data[qid]['answer']
    full_answer = data[qid]['fullAnswer']
    entailed = data[qid]['entailed']
    return image, question, answer, full_answer, entailed

questions = {}
answers = {}
ground_truth = {}
full_ground_truth = {}

with open("questions.pkl", 'rb') as file:
    questions = pickle.load(file)

with open("answers.pkl", 'rb') as file:
    answers = pickle.load(file)

with open("ground_truth.pkl", 'rb') as file:
    ground_truth = pickle.load(file)

with open("full_ground_truth.pkl", 'rb') as file:
    full_ground_truth = pickle.load(file)

print(len(questions), file=sys.stderr)
print(len(answers), file=sys.stderr)
print(len(ground_truth), file=sys.stderr)
print(len(full_ground_truth), file=sys.stderr)

print(len(new_data), file=sys.stderr)
tic = time()
toc = time()
print(f'Time elapsed: {toc-tic:0.2f} seconds\n', file=sys.stderr)
for i, k in enumerate(list(new_data.keys())):
    if (i+1) % 10000 == 0:
        toc = time()
        print(f'Time elapsed: {toc-tic:0.2f} seconds\n', file=sys.stderr)
        tic = time()
        print(i+1, file=sys.stderr)
        dictionaries = {'questions.pkl': questions, 'answers.pkl': answers, 
                'ground_truth.pkl': ground_truth, 'full_ground_truth.pkl': full_ground_truth}

        for filename, dictionary in dictionaries.items():
            with open(filename, 'wb') as file:
                pickle.dump(dictionary, file)

    if k not in questions:
        image, question, answer, full_answer, entailed = get_data(k, new_data)

        question = f"{question} Answer simply with one word:"
        try:
            answer = word_tokenize(PromptPipeline(question, f"/scratch/chaijy_root/chaijy0/josuetf/mac-network/data/images/{image}.jpg")[0].split(":")[1])[0]
        except:
            print("Answer couldn't be generated", file=sys.stderr)
            answer = ""
        answers[k] = answer.lower()
        questions[k] = question
        ground_truth[k] = answer
        full_ground_truth[k] = full_answer

dictionaries = {'questions.pkl': questions, 'answers.pkl': answers, 
                'ground_truth.pkl': ground_truth, 'full_ground_truth.pkl': full_ground_truth}

for filename, dictionary in dictionaries.items():
    with open(filename, 'wb') as file:
        pickle.dump(dictionary, file)

with open("questions.json", "w") as f:
    json.dump(questions, f)

print("DONE", file=sys.stderr)
        

