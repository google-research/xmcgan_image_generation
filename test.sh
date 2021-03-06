# Copyright 2021 The XMC-GAN Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
CONFIG="xmcgan/configs/coco_xmc.py"
EXP_NAME=$1
WORKDIR="/path/to/exp/$EXP_NAME"  # CHANGEME

CUDA_VISIBLE_DEVICES="7" python -m xmcgan.main \
  --config="$CONFIG" \
  --mode="test" \
  --workdir="$WORKDIR" \
