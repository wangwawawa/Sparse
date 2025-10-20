## Cognitive Contour Detection of Sparse-Structured Objects in the Alpha-Shape Scale Space

This paper proposes a novel scale-space method for detecting cognitive contours of sparse-structured objects. Additionally, a benchmark image dataset of sparse-structured objects is constructed in this paper, which is publicly available for evaluating cognitive contour detection performance. Here are the code and the dataset for this paper.

### Dateset

Download the SOIS dataset from [BaiDuNetdisk](https://pan.baidu.com/s/1SaOsQ61qiwma0HuqM9eclQ&pwd=inss) or [Google Drive](https://drive.google.com/file/d/1cgXnLPr1eEML67M4IeR0SvpI-MDxxA7D/view?usp=sharing).

### Requirements

* Python
  * scipy
  * numba
  * cv2
  * copy

### How to use

* This demo can be easily used by running the 'demo.py' file.

  ```
  python demo.py
  ```

* To extract inner and outer contours with the anisotropic gradient-based
  operator introduced in this repository, run ``gradient_contours.py`` and pass
  the path to the input image. Intermediate results and the final overlay are
  written to ``data/output`` by default.

  ```bash
  python gradient_contours.py data/input/16.png --output data/output
  ```

  The script uses multi-directional first- and second-order anisotropic
  Gaussian derivatives to estimate gradient orientation and strength, applies
  non-maximum suppression, and performs adaptive thresholding followed by
  morphological post-processing to recover both the outer (red) and inner
  (green) contours of the sparse structure.

### Citation

If you find this work helpful for your research, we would be grateful if you could cite:

```bibtex
@article{shen2025cognitive,
  title={Cognitive Contour Detection of Sparse-Structured Objects in the Alpha-Shape Scale Space},
  author={Shen, Yuxiang and Zhong, Baojiang and Ma, Kai-Kuang},
  journal={IEEE Transactions on Image Processing},
  year={2025},
  publisher={IEEE}
}
```
