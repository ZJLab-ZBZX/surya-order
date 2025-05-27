# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 11:42:10 2024

@author: ZJ
"""
from transformers import DetrConfig, BeitConfig, DetrImageProcessor, VisionEncoderDecoderConfig, AutoModelForCausalLM, \
    AutoModel
from modules.surya_order.config import MBartOrderConfig, VariableDonutSwinConfig
from modules.surya_order.decoder import MBartOrder
from modules.surya_order.encoder import VariableDonutSwinModel
from modules.surya_order.encoderdecoder import OrderVisionEncoderDecoderModel
from modules.surya_order.processor import OrderImageProcessor
from modules.surya_order.settings import settings
from modules.surya_order.schema import OrderBox, OrderResult
from transformers import DonutImageProcessor, DonutProcessor
from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import PILImageResampling, ImageInput, ChannelDimension, make_list_of_images, \
    valid_images, to_numpy_array
from copy import deepcopy
from PIL import Image
import torch
import numpy as np
import time, os

def batch_ordering(model, processor, bboxes, orig_sizes, batch_pixel_values, batch_bboxes, batch_bbox_mask, batch_bbox_counts):
    token_count = 0
    encoder_outputs = None
    past_key_values = None
    batch_predictions = [[] for _ in range(len(batch_pixel_values))]
    done = torch.zeros(len(batch_pixel_values), dtype=torch.bool, device=model.device)
    output_order = []
    with torch.inference_mode():
        while token_count < settings.ORDER_MAX_BOXES: 
            print('='*50)
            print('batch_pixel_values', batch_pixel_values.shape)
            print('batch_bboxes', batch_bboxes.shape)
            print('batch_bbox_mask', batch_bbox_mask.shape)
            print('batch_bbox_counts', batch_bbox_counts.shape)
            if encoder_outputs is not None:
                print('encoder_outputs', len(encoder_outputs), encoder_outputs[0].shape)
            if past_key_values is not None:
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

            last_tokens = []
            last_token_mask = []
            min_val = torch.finfo(model.dtype).min
            for j in range(logits.shape[0]):
                label_count = batch_bbox_counts[j, 1] - batch_bbox_counts[j, 0] - 1  # Subtract 1 for the sep token
                new_logits = logits[j, -1]
                new_logits[batch_predictions[j]] = min_val  # Mask out already predicted tokens, we can only predict each token once
                new_logits[label_count:] = min_val  # Mask out all logit positions above the number of bboxes
                pred = int(torch.argmax(new_logits, dim=-1).item())

                # Add one to avoid colliding with the 1000 height/width token for bboxes
                last_tokens.append([[pred + processor.box_size["height"] + 1] * 4])
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

            batch_bboxes = torch.tensor(last_tokens, dtype=torch.long).to(model.device)
            token_bbox_mask = torch.tensor(last_token_mask, dtype=torch.long).to(model.device)
            batch_bbox_mask = torch.cat([batch_bbox_mask, token_bbox_mask], dim=1)
            token_count += 1

    for j, row_pred in enumerate(batch_predictions):
        row_bboxes = bboxes[j]
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


def batch_ordering_om(om_session_1, om_session_2, processor, bboxes, orig_sizes, 
                        batch_pixel_values, batch_bboxes, batch_bbox_mask, batch_bbox_counts):    
    token_count = 0
    encoder_outputs = None
    batch_predictions = [[] for _ in range(len(batch_pixel_values))]
    done = np.zeros(len(batch_pixel_values), dtype=bool)
    past_key_values_0_0 = past_key_values_0_1 = past_key_values_0_2 = past_key_values_0_3 = None
    past_key_values_1_0 = past_key_values_1_1 = past_key_values_1_2 = past_key_values_1_3 = None
    past_key_values_2_0 = past_key_values_2_1 = past_key_values_2_2 = past_key_values_2_3 = None
    num_boxes = batch_bboxes.shape[1]        
    output_order = []
    while token_count < settings.ORDER_MAX_BOXES:
        if encoder_outputs is None:
            custom_sizes = 4*(1024*1024+(4*1024*64*2+4*num_boxes*64*2)*4+num_boxes*261)
            outputs = om_session_1.infer([
                batch_pixel_values, 
                batch_bboxes, 
                batch_bbox_mask, 
                batch_bbox_counts],
                mode='dymshape',
                custom_sizes=custom_sizes               
                )
            logits = outputs[0].reshape([1, num_boxes, 261])
        else:
            custom_sizes = 4*(1024*1024+(4*1024*64*2+4*num_boxes*64*2)*4+261)
            outputs = om_session_2.infer([
                batch_bboxes,
                batch_bbox_mask, 
                encoder_outputs,
                past_key_values_0_0, 
                past_key_values_0_1, 
                past_key_values_0_2, 
                past_key_values_0_3,
                past_key_values_1_0, 
                past_key_values_1_1, 
                past_key_values_1_2, 
                past_key_values_1_3,
                past_key_values_2_0, 
                past_key_values_2_1,
                past_key_values_2_2, 
                past_key_values_2_3],
                mode='dymshape',
                custom_sizes=12693744 
                ) 
            logits = outputs[0].reshape([1, 1, 261])
        # print(logits)
        past_key_values_0_0, past_key_values_0_1 =  outputs[1], outputs[2]
        past_key_values_0_2, past_key_values_0_3 =  outputs[3], outputs[4]
        past_key_values_1_0, past_key_values_1_1 =  outputs[5], outputs[6]
        past_key_values_1_2, past_key_values_1_3 =  outputs[7], outputs[8]
        past_key_values_2_0, past_key_values_2_1 =  outputs[9], outputs[10]
        past_key_values_2_2, past_key_values_2_3 =  outputs[11], outputs[12]
        encoder_outputs = outputs[13]    
        last_tokens = []
        last_token_mask = []
        min_val = np.finfo(np.float32).tiny
        for j in range(logits.shape[0]):
            label_count = batch_bbox_counts[j, 1] - batch_bbox_counts[j, 0] - 1  # Subtract 1 for the sep token
            new_logits = logits[j, -1]
            new_logits[batch_predictions[j]] = min_val  # Mask out already predicted tokens, we can only predict each token once
            new_logits[label_count:] = min_val  # Mask out all logit positions above the number of bboxes
            pred = int(np.argmax(new_logits, axis=-1))

            # Add one to avoid colliding with the 1000 height/width token for bboxes
            last_tokens.append([[pred + processor.box_size["height"] + 1] * 4])
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
        batch_bboxes = np.array(last_tokens, dtype=np.int64)
        token_bbox_mask = np.array(last_token_mask)
        batch_bbox_mask = np.concatenate([batch_bbox_mask, token_bbox_mask], axis=1).astype(np.int64)
        token_count += 1
        num_boxes += 1

    for j, row_pred in enumerate(batch_predictions):
        row_bboxes = bboxes[j]
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


class SuryaOrder():
    def __init__(self, checkpoint='models/vikp/surya_order', device='cuda'):
        self.device = device
        if device == 'npu':        
            from ais_bench.infer.interface import InferSession
            device_id = 0
            self.om_session_1 = InferSession(int(device_id), os.path.join(checkpoint, "surya_order_1.om"))  
            self.om_session_2 = InferSession(int(device_id), os.path.join(checkpoint, "surya_order_2.om")) 
        else:
            dtype=settings.MODEL_DTYPE 
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
            model = model.to(device)
            model = model.eval()
            self.model = model
        processor = OrderImageProcessor.from_pretrained(checkpoint)
        processor.size = settings.ORDER_IMAGE_SIZE
        box_size = 1024
        max_tokens = 256
        processor.token_sep_id = max_tokens + box_size + 1
        processor.token_pad_id = max_tokens + box_size + 2
        processor.max_boxes = settings.ORDER_MAX_BOXES - 1
        processor.box_size = {"height": box_size, "width": box_size}
        self.processor = processor 
    
    def predict(self, layout_rects, pil_img):
        
        images = [pil_img]
        bboxes = [layout_rects.tolist()]
        images = [image.convert("RGB") for image in images] # also copies the images
        batch_size = 1
        i = 0
        batch_bboxes = deepcopy(bboxes[i:i+batch_size]) 
        batch_images = images[i:i+batch_size] 
        orig_sizes = [image.size for image in batch_images]
        model_inputs = self.processor(images=batch_images, boxes=batch_bboxes)

        batch_pixel_values = model_inputs["pixel_values"]
        batch_bboxes = model_inputs["input_boxes"]
        batch_bbox_mask = model_inputs["input_boxes_mask"]
        batch_bbox_counts = model_inputs["input_boxes_counts"]

        batch_bboxes = torch.from_numpy(np.array(batch_bboxes, dtype=np.int32))
        batch_bbox_mask = torch.from_numpy(np.array(batch_bbox_mask, dtype=np.int32))
        batch_pixel_values = torch.tensor(np.array(batch_pixel_values), dtype=torch.float) # dtype=model.dtype
        batch_bbox_counts = torch.tensor(np.array(batch_bbox_counts), dtype=torch.long)

        if self.device == 'npu':           
            t1 = time.time()
            layout_order = np.argsort([res['position'] for res in batch_ordering_om(
                self.om_session_1, self.om_session_2, self.processor, bboxes, orig_sizes,
                batch_pixel_values, batch_bboxes, batch_bbox_mask, batch_bbox_counts)[0].dict()['bboxes']]) 
            print('ordering time', time.time() - t1)
        else:
            batch_bboxes = batch_bboxes.to(self.model.device)
            batch_bbox_mask = batch_bbox_mask.to(self.model.device)
            batch_pixel_values = batch_pixel_values.to(self.model.device)
            batch_bbox_counts = batch_bbox_counts.to(self.model.device)
            layout_order = np.argsort([res['position'] for res in batch_ordering(
                self.model, self.processor, bboxes, orig_sizes, batch_pixel_values, batch_bboxes, batch_bbox_mask, batch_bbox_counts)[0].dict()['bboxes']])           
        return layout_order


if __name__ == '__main__': 
    layout_rects = np.load('surya/layout_rects2.npy') 
    pil_img = Image.open('surya/pil_img2.jpg') 
    surya_model = SuryaOrder(checkpoint='models/vikp/surya_order', device='npu') 
    t1 = time.time()
    res = surya_model.predict(layout_rects, pil_img)
    print(time.time() - t1)
    print(res)









