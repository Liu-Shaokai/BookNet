# <div align="center">BookNet: Dual-Page Book Image Rectification <br/>via Cross-Page Attention</div>

<div align="center">
  <a href="https://arxiv.org/abs/2601.21938"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2601.21938-CE0000?logo=arXiv&logoColor=white&labelColor=gray"></a>
  <a href="https://huggingface.co/datasets/skliu520/Book100"><img alt="HuggingFace Dataset" src="https://img.shields.io/badge/🤗_HuggingFace-Book100-FFBF00?labelColor=gray"></a>
  <a href="./LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-008844?logo=opensourceinitiative&logoColor=white&labelColor=gray"></a>
</div>

Official implementation of [**Dual-Page Book Image Rectification via Cross-Page Attention**](https://arxiv.org/abs/2601.21938).

BookNet rectifies photographed dual-page book images by explicitly modeling the geometric correlation between the two facing pages with a cross-page attention mechanism. We also release **Book3D**, a large-scale synthetic training set of dual-page book distortions, and **Book100**, a real-world benchmark for evaluation.

## Model

The model is defined in [`model.py`](./model.py) in the root directory.

## Datasets

| Dataset | Type | Link |
| --- | --- | --- |
| Book3D | Synthetic training set | [Baidu Netdisk](https://pan.baidu.com/s/1aBL9MYqcRviWMeErxEYMBA?pwd=5hcf) (code: `5hcf`) |
| Book100 | Real-world benchmark | [Hugging Face](https://huggingface.co/datasets/skliu520/Book100/) |

> ⚠️ **Note.** Book3D and Book100 are released **for non-commercial scientific research purposes only**. The book pages contained in these datasets may be subject to third-party copyright, and the datasets must not be used for any commercial purpose or redistributed for commercial use. By downloading them you agree to these terms.

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

## License

The code in this repository is released under the [MIT License](./LICENSE). Note that the license covers the code only; the accompanying datasets (Book3D and Book100) are provided for non-commercial academic research use, as stated above.

## Acknowledgement

The codes are largely based on [DocUNet](https://www3.cs.stonybrook.edu/~cvl/docunet.html), [DewarpNet](https://github.com/cvlab-stonybrook/DewarpNet), [UVDoc](https://github.com/tanguymagne/UVDoc) and [PaddleOCR-VL](https://github.com/PaddlePaddle/PaddleOCR/). Thanks for their wonderful works.
