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

* To obtain the gradient-driven binary mask together with the contour recovered
  through the sparse triangulation pipeline, run ``gradient_contours.py`` and
  pass the path to the input image. The script stores exactly two artefacts in
  ``data/output`` by default: ``*_contour.png`` (a one-pixel outline extracted
  from the final binary mask) and ``*_binary.png`` (the cleaned binary mask
  inferred from anisotropic gradient responses and refined by ``sparse.edge``).

  ```bash
  python gradient_contours.py data/input/16.png --output data/output
  ```

  The workflow mirrors the original repository: multi-directional first- and
  second-order anisotropic Gaussian derivatives provide dense gradient
  responses; non-maximum suppression and adaptive thresholding with an
  additional Otsu prior generate a confident binary mask of the swarm region.
  This mask is handed to the existing ``sparse.edge`` routine, which performs
  the Delaunay-based contour reconstruction. The resulting binary mask is then
  used both as the saved mask and to trace the outline, ensuring the contour
  and binary outputs remain pixel-wise consistent. Only the contour and binary
  images are saved to disk, making it straightforward to evaluate segmentation
  accuracy.

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
