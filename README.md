# BookNet

Official implementation of **Dual-Page Book Image Rectification via Cross-Page Attention**.

BookNet rectifies photographed dual-page book images by explicitly modeling the geometric correlation between the two facing pages with a cross-page attention mechanism. We also release **Book3D**, a large-scale synthetic training set of dual-page book distortions, and **Book100**, a real-world benchmark for evaluation.

## Model

The model is defined in [`model.py`](./model.py) in the root directory.

## Datasets

| Dataset | Type | Link |
| --- | --- | --- |
| Book3D | Synthetic training set | [Baidu Netdisk](https://pan.baidu.com/s/1aBL9MYqcRviWMeErxEYMBA?pwd=5hcf) (code: `5hcf`) |
| Book100 | Real-world benchmark | [Hugging Face](https://huggingface.co/datasets/skliu520/Book100/) |

## Eval

Please refer to the evaluation code in the [`eval`](./eval) folder.

## Citation

```bibtex
@article{liu2026booknet,
  title={BookNet: Book Image Rectification via Cross-Page Attention Network},
  author={Liu, Shaokai and Feng, Hao and Luan, Bozhi and Hou, Min and Deng, Jiajun and Zhou, Wengang},
  journal={arXiv preprint arXiv:2601.21938},
  year={2026}
}
```

## Acknowledgement

The codes are largely based on [DocUNet](https://www3.cs.stonybrook.edu/~cvl/docunet.html), [DewarpNet](https://github.com/cvlab-stonybrook/DewarpNet), [UVDoc](https://github.com/tanguymagne/UVDoc) and [PaddleOCR-VL](https://github.com/PaddlePaddle/PaddleOCR/). Thanks for their wonderful works.