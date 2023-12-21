# Copyright 2021 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch

from typing import Union, Any
from torchvision import transforms
from towhee.operator.base import NNOperator
from towhee import register
from PIL import Image as PILImage
from model.feature_extractor import FeatureExtractor
from model.students import FineGrainedStudent, CoarseGrainedStudent
from model.selector import SelectorNetwork
from torch import nn


@register(output_schema=['vec'])
class DistillAndSelect(NNOperator):
    """
    DistillAndSelect
    """

    def __init__(self, model_name: str, model_weight_path: str = None,
                 feature_extractor: Union[str, nn.Module] = 'default', device: str = None):
        """

        Args:
            model_name (`str`):
                Can be one of them:
                `feature_extractor`: Feature Extractor only,
                `fg_att_student`: Fine Grained Student with attention,
                `fg_bin_student`: Fine Grained Student with binarization,
                `cg_student`: Coarse Grained Student,
                `selector_att`: Selector Network with attention,
                `selector_bin`: Selector Network with binarization.
            model_weight_path (`str`):
                Default is None, download use the original pretrained weights.
            feature_extractor (`Union[str, nn.Module]`):
                `None`, 'default' or a pytorch nn.Module instance.
                `None` means this operator don't support feature extracting from the video data and this operator process embedding feature as input.
                'default' means using the original pretrained feature extracting weights and this operator can process video data as input.
                Or you can pass in a nn.Module instance as a specific feature extractor.
                Default is `default`.
            device (`str`):
                Model device, cpu or cuda.
        """
        super().__init__()
        assert model_name in ['feature_extractor', 'fg_att_student', 'fg_bin_student', 'cg_student', 'selector_att',
                              'selector_bin'], 'unsupported model.'
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_name = model_name

        self.feature_extractor = None
        if feature_extractor == 'default':
            self.feature_extractor = FeatureExtractor(dims=512).to(device).eval()
        elif isinstance(feature_extractor, nn.Module):
            self.feature_extractor = feature_extractor

        self.model = None
        pretraind = True if model_weight_path is None else None
        if self.model_name == 'fg_att_student':
            self.model = FineGrainedStudent(pretrained=pretraind, attention=True)
        elif self.model_name == 'fg_bin_student':
            self.model = FineGrainedStudent(pretrained=pretraind, binarization=True)

        elif self.model_name == 'cg_student':
            self.model = CoarseGrainedStudent(pretrained=pretraind)

        elif self.model_name == 'selector_att':
            self.model = SelectorNetwork(pretrained=pretraind, attention=True)
        elif self.model_name == 'selector_bin':
            self.model = SelectorNetwork(pretrained=pretraind, binarization=True)

        if model_weight_path is not None:
            self.model.load_state_dict(torch.load(model_weight_path))

        if self.model is not None:
            self.model.to(device).eval()

        self.tfms = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])

    def __call__(self, data: Any):  # List[VideoFrame] when self.feature_extractor is not None
        if self.feature_extractor is not None:
            pil_img_list = []
            for img in data:
                pil_img = PILImage.fromarray(img, img.mode)
                tfmed_img = self.tfms(pil_img).permute(1, 2, 0).unsqueeze(0)
                pil_img_list.append(tfmed_img)
            data = torch.concat(pil_img_list, dim=0) * 255
            data = self.feature_extractor(data.to(self.device)).to(self.device)
            if self.model_name == 'feature_extractor':
                return data.cpu().detach().squeeze().numpy()
        index_feature = self.model.index_video(data)
        return index_feature.cpu().detach().squeeze().numpy()
