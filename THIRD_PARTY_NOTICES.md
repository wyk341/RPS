# Third-party notices

RPS includes adaptations of the following projects:

- [MTL-TabNet](https://github.com/namtuanly/MTL-TabNet): table recognition model components and data representation.
- [TableMASTER-mmocr](https://github.com/JiaquanYe/TableMASTER-mmocr): table structure convertors, positional encoding, and decoder components.
- [MMOCR](https://github.com/open-mmlab/mmocr) and [MMDetection](https://github.com/open-mmlab/mmdetection): recognition components, residual visual backbone, and supporting neural-network utilities. The upstream Apache License 2.0 text is provided in `licenses/Apache-2.0.txt`.
- [Ranger](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer): the optimizer implementation combines RAdam, Lookahead, and Gradient Centralization. The source credit is retained in `rps/optim.py`, and the upstream license is in `licenses/Ranger-Apache-2.0.txt`.
- [PubTabNet](https://github.com/ibm-aur-nlp/PubTabNet): structure-only tree-edit evaluation conventions.

The dependencies named in `requirements.txt` retain their respective upstream licenses.

Model, convertor, loss, and metric modules have been adapted to the `rps` package layout with framework-independent construction and imports. The upstream MMOCR license text retained by MTL-TabNet is provided in `licenses/MMOCR-Apache-2.0.txt`.
