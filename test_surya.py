# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 11:42:10 2024

@author: ZJ
"""
# from surya.model.ordering.processor import load_processor as surya_load_processor
# from surya.model.ordering.model import load_model as surya_load_model
# from surya.ordering import batch_ordering
from transformers import DetrConfig, BeitConfig, DetrImageProcessor, VisionEncoderDecoderConfig, AutoModelForCausalLM, \
    AutoModel
from modules.surya_order.config import MBartOrderConfig, VariableDonutSwinConfig
from modules.surya_order.decoder import MBartOrder
from modules.surya_order.encoder import VariableDonutSwinModel
from modules.surya_order.encoderdecoder import OrderVisionEncoderDecoderModel
from modules.surya_order.processor import OrderImageProcessor
from transformers import DonutImageProcessor, DonutProcessor
from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import PILImageResampling, ImageInput, ChannelDimension, make_list_of_images, \
    valid_images, to_numpy_array
from modules.surya_order.settings import settings
from modules.surya_order.schema import OrderBox, OrderResult
from copy import deepcopy
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm

import pickle

def get_batch_size():
    batch_size = settings.ORDER_BATCH_SIZE
    if batch_size is None:
        batch_size = 8
        if settings.TORCH_DEVICE_MODEL == "mps":
            batch_size = 8
        if settings.TORCH_DEVICE_MODEL == "cuda":
            batch_size = 32
    return batch_size

def batch_ordering(model, device, batch_pixel_values, batch_bboxes, batch_bbox_mask, batch_bbox_counts):
    token_count = 0
    encoder_outputs = None
    past_key_values = None
    batch_predictions = [[] for _ in range(len(batch_images))]
    done = torch.zeros(len(batch_images), dtype=torch.bool, device=device)

 #   with open('testdata.pkl', 'wb') as fid:
 #       pickle.dump([batch_pixel_values, batch_bboxes, batch_bbox_mask, batch_bbox_counts], fid)
 #   exit()
    print('='*50)
    print('batch_pixel_values', batch_pixel_values.shape, batch_pixel_values.dtype)
    print('batch_bboxes', batch_bboxes.shape, batch_bboxes.dtype)
    print('batch_bbox_mask', batch_bbox_mask.shape, batch_bbox_mask.dtype)
    print('batch_bbox_counts', batch_bbox_counts.shape, batch_bbox_counts.dtype)
#    print(batch_bboxes)
#    print(batch_bbox_mask)
    print(batch_bbox_counts)


    with torch.inference_mode():
        while token_count < settings.ORDER_MAX_BOXES: 
            print('='*50)
            print('batch_pixel_values', batch_pixel_values.shape)
            print('batch_bboxes', batch_bboxes.shape)
            print('batch_bbox_mask', batch_bbox_mask.shape)
            print('batch_bbox_counts', batch_bbox_counts.shape)
            print(f"TOKEN_IDX: {token_count}")
            print(f'!!!!!!!!!! encoder_outputs {type(encoder_outputs)}')

            if encoder_outputs is not None:
                print('\t\t???? encoder_outputs')
                print('encoder_outputs', len(encoder_outputs), encoder_outputs[0].shape)
            print(f'!!!!!!!!!! past_key_values {type(past_key_values)}')
            if past_key_values is not None:
                print('???????????? past_key_values')
                print('past_key_values', len(past_key_values), len(past_key_values[0]), past_key_values[0][0].shape, past_key_values[0][1].shape, 
                  past_key_values[1][0].shape, past_key_values[1][1].shape)         
            return_dict = model(
                pixel_values=batch_pixel_values,
                decoder_input_boxes=batch_bboxes,
                decoder_input_boxes_mask=batch_bbox_mask,
                decoder_input_boxes_counts=batch_bbox_counts,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
            )
            logits = return_dict["logits"].detach()
            print('++++++++++++')
            print(return_dict.keys())
            print(logits.shape)
            print(logits[0, -1].shape)
            print('batch_predictions', type(batch_predictions))
            print('logic: ', logits.shape)
            print("past_key_values", type(return_dict["past_key_values"]))
    
            last_tokens = []
            last_token_mask = []
            min_val = torch.finfo(model.dtype).min
            for j in range(logits.shape[0]):
                label_count = batch_bbox_counts[j, 1] - batch_bbox_counts[j, 0] - 1  # Subtract 1 for the sep token
                new_logits = logits[j, -1] # logits: 2 * 12 * 261 
                new_logits[batch_predictions[j]] = min_val  # Mask out already predicted tokens, we can only predict each token once
                new_logits[label_count:] = min_val  # Mask out all logit positions above the number of bboxes
                pred = int(torch.argmax(new_logits, dim=-1).item())

                # Add one to avoid colliding with the 1000 height/width token for bboxes
                last_tokens.append([[pred + processor.box_size["height"] + 1] * 4])
                print('!!!!!!!!!!!!!=============', processor.box_size["height"])
                if len(batch_predictions[j]) == label_count - 1:  # Minus one since we're appending the final label
                    last_token_mask.append([0])
                    batch_predictions[j].append(pred)
                    done[j] = True
                elif len(batch_predictions[j]) < label_count - 1:
                    last_token_mask.append([1])
                    batch_predictions[j].append(pred)  # Get rank prediction for given position
                else:
                    last_token_mask.append([0])

            if done.all():
                break

            past_key_values = return_dict["past_key_values"]
            encoder_outputs = (return_dict["encoder_last_hidden_state"],)

            batch_bboxes = torch.tensor(last_tokens, dtype=torch.long).to(device)
            token_bbox_mask = torch.tensor(last_token_mask, dtype=torch.long).to(device)
            batch_bbox_mask = torch.cat([batch_bbox_mask, token_bbox_mask], dim=1)
            token_count += 1

    for j, row_pred in enumerate(batch_predictions):
        row_bboxes = bboxes[i+j]
        assert len(row_pred) == len(row_bboxes), f"Mismatch between logits and bboxes. Logits: {len(row_pred)}, Bboxes: {len(row_bboxes)}"

        orig_size = orig_sizes[j]
        ranks = [0] * len(row_bboxes)

        for box_idx in range(len(row_bboxes)):
            ranks[row_pred[box_idx]] = box_idx

        order_boxes = []
        for row_bbox, rank in zip(row_bboxes, ranks):
            order_box = OrderBox(
                bbox=row_bbox,
                position=rank,
            )
            order_boxes.append(order_box)

        result = OrderResult(
            bboxes=order_boxes,
            image_bbox=[0, 0, orig_size[0], orig_size[1]],
        )
        output_order.append(result)
    return output_order



dtype=settings.MODEL_DTYPE
device = 'cpu'
checkpoint = '../models/vikp/surya_order'


config = VisionEncoderDecoderConfig.from_pretrained(checkpoint)

decoder_config = vars(config.decoder)
decoder = MBartOrderConfig(**decoder_config)
config.decoder = decoder
encoder_config = vars(config.encoder)
encoder = VariableDonutSwinConfig(**encoder_config) 
config.encoder = encoder
# Get transformers to load custom model
AutoModel.register(MBartOrderConfig, MBartOrder)
AutoModelForCausalLM.register(MBartOrderConfig, MBartOrder)
AutoModel.register(VariableDonutSwinConfig, VariableDonutSwinModel)
model = OrderVisionEncoderDecoderModel.from_pretrained(checkpoint, config=config, torch_dtype=dtype)
assert isinstance(model.decoder, MBartOrder)
assert isinstance(model.encoder, VariableDonutSwinModel)
model = model.to(device)
model = model.eval()

processor = OrderImageProcessor.from_pretrained(checkpoint)
processor.size = settings.ORDER_IMAGE_SIZE
box_size = 1024
max_tokens = 256
processor.token_sep_id = max_tokens + box_size + 1
processor.token_pad_id = max_tokens + box_size + 2
processor.max_boxes = settings.ORDER_MAX_BOXES - 1
processor.box_size = {"height": box_size, "width": box_size}

layout_rects = np.load('surya/layout_rects.npy')
pil_img = Image.open('surya/pil_img.jpg')
layout_rects2 = np.load('surya/layout_rects2.npy')
pil_img2 = Image.open('surya/pil_img2.jpg')

images = [pil_img, pil_img2]
bboxes = [layout_rects.tolist(), layout_rects2.tolist()]

images = [image.convert("RGB") for image in images] # also copies the images
batch_size = 2
i = 0
output_order = []
batch_bboxes = deepcopy(bboxes[i:i+batch_size]) 
batch_images = images[i:i+batch_size] 
orig_sizes = [image.size for image in batch_images]
model_inputs = processor(images=batch_images, boxes=batch_bboxes)

batch_pixel_values = model_inputs["pixel_values"]
batch_bboxes = model_inputs["input_boxes"]
batch_bbox_mask = model_inputs["input_boxes_mask"]
batch_bbox_counts = model_inputs["input_boxes_counts"]

batch_bboxes = torch.from_numpy(np.array(batch_bboxes, dtype=np.int32)).to(model.device)
batch_bbox_mask = torch.from_numpy(np.array(batch_bbox_mask, dtype=np.int32)).to(model.device)
batch_pixel_values = torch.tensor(np.array(batch_pixel_values), dtype=model.dtype).to(model.device)
batch_bbox_counts = torch.tensor(np.array(batch_bbox_counts), dtype=torch.long).to(model.device)

order_results =  batch_ordering(
    model, device, batch_pixel_values, batch_bboxes, batch_bbox_mask, batch_bbox_counts)
print(order_results)

layout_order = [np.argsort([r['position'] for r in res.dict()['bboxes']]) for res in order_results]
print(layout_order)