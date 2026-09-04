# Third-party software and model notices

NaturalLab's own source code is distributed under the MIT License. That license
does not replace the licenses or usage terms of dependencies, services, or
model weights selected by a researcher.

## Qwen3.6-27B service preset

NaturalLab ships a client and configuration preset for the separately deployed
[`Qwen/Qwen3.6-27B` model](https://huggingface.co/Qwen/Qwen3.6-27B). The
official model repository identifies the model as Apache-2.0 licensed. No Qwen
weights are included in this repository or in NaturalLab distribution
artifacts. Review the model repository and the terms of the service used to
host it before deployment.

## Optional Ultralytics YOLO path

NaturalLab can call the separately installed `ultralytics` package and YOLO
checkpoints when the `yolo` extra is selected. Ultralytics currently describes
its software licensing options as AGPL-3.0 and an Enterprise license. Installing
or using this optional path may therefore create obligations beyond
NaturalLab's MIT License, particularly for distribution or deployed services.
Review the [official Ultralytics licensing information](https://github.com/ultralytics/ultralytics#license)
and the terms for the exact package and weights before use. This repository
does not grant an Ultralytics Enterprise license.

## OSNet person ReID

NaturalLab includes an adapted OSNet implementation carrying the original
Kaiyang Zhou MIT notice and can download one checksum-pinned OSNet-AIN x1.0
MSMT17 checkpoint. The upstream
[`deep-person-reid` repository](https://github.com/KaiyangZhou/deep-person-reid)
and [`kaiyangzhou/osnet` model repository](https://huggingface.co/kaiyangzhou/osnet)
identify those materials as MIT-licensed. Keep the embedded copyright notice
when redistributing the adapted implementation.

Dependency package metadata and model/service terms are authoritative. This
notice highlights integrations most likely to surprise a user; it is not legal
advice or a complete substitute for reviewing the dependencies used in a
particular installation.
