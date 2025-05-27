# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:05:57 2024

@author: ZJ
"""
from typing import List, Tuple, Any, Optional
import copy
from pydantic import BaseModel, field_validator, computed_field

def rescale_bbox(bbox, processor_size, image_size):
    page_width, page_height = processor_size

    img_width, img_height = image_size
    width_scaler = img_width / page_width
    height_scaler = img_height / page_height

    new_bbox = copy.deepcopy(bbox)
    new_bbox[0] = int(new_bbox[0] * width_scaler)
    new_bbox[1] = int(new_bbox[1] * height_scaler)
    new_bbox[2] = int(new_bbox[2] * width_scaler)
    new_bbox[3] = int(new_bbox[3] * height_scaler)
    return new_bbox


class PolygonBox(BaseModel):
    polygon: List[List[float]]
    confidence: Optional[float] = None

    @field_validator('polygon')
    @classmethod
    def check_elements(cls, v: List[List[float]]) -> List[List[float]]:
        if len(v) != 4:
            raise ValueError('corner must have 4 elements')

        for corner in v:
            if len(corner) != 2:
                raise ValueError('corner must have 2 elements')
        return v


class Bbox(BaseModel):
    bbox: List[float]

    @field_validator('bbox')
    @classmethod
    def check_4_elements(cls, v: List[float]) -> List[float]:
        if len(v) != 4:
            raise ValueError('bbox must have 4 elements')
        return v

    def rescale_bbox(self, orig_size, new_size):
        self.bbox = rescale_bbox(self.bbox, orig_size, new_size)

    def round_bbox(self, divisor):
        self.bbox = [x // divisor * divisor for x in self.bbox]

    @property
    def height(self):
        return self.bbox[3] - self.bbox[1]

    @property
    def width(self):
        return self.bbox[2] - self.bbox[0]

    @property
    def area(self):
        return self.width * self.height

    @property
    def polygon(self):
        return [[self.bbox[0], self.bbox[1]], [self.bbox[2], self.bbox[1]], [self.bbox[2], self.bbox[3]], [self.bbox[0], self.bbox[3]]]

    
class OrderBox(Bbox):
    position: int
    
class TextLine(PolygonBox):
    text: str
    confidence: Optional[float] = None
    
class OCRResult(BaseModel):
    text_lines: List[TextLine]
    languages: List[str]
    image_bbox: List[float]

class OrderResult(BaseModel):
    bboxes: List[OrderBox]
    image_bbox: List[float]