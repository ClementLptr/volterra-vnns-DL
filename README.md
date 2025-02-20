# Volterra Neural Networks (VNNs) - RKHS Implementation  

This repository extends and improves the codebase for [Volterra Neural Networks (VNNs)](https://arxiv.org/abs/1910.09616) and [Conquering the CNN Over-Parameterization Dilemma](https://ojs.aaai.org/index.php/AAAI/article/view/6870) by integrating **Reproducing Kernel Hilbert Space (RKHS)** representations to enhance Volterra filter implementations.  

As part of a research project at **CentraleSupélec**, this work explores and extends the theoretical and practical applications of VNNs within the RKHS framework. This approach provides a mathematically rigorous way to generalize Volterra filters, with potential improvements in **accuracy, efficiency, and model interpretability**.  

---

## Key Features  

✅ **RKHS-Based Volterra Filters** – Novel implementation embedding Volterra filters into the RKHS framework.  
✅ **Flexible Model Configurations** – Support for **RGB, optical flow, and complex inputs** with adaptable architectures.  
✅ **Improved Computational Efficiency** – Real-time computation of optical flow, with optional **pre-computation for faster training**.  
✅ **One-Time Video Preprocessing** – Video frames are preprocessed once during the initial setup.  

---

## Getting Started  

### Prerequisites  

Ensure you have **Python 3.x** installed and install dependencies from `requirements.txt`:  
```bash
pip install -r requirements.txt
```

### Training  

1. **Set Dataset Paths** – Update `mypath.py` with the correct dataset locations.  
2. **Choose Model Architecture** – Configure your model in `train_VNN_fusion_highQ.py`. For **RKHS-based models**, use `networks/vnn_rkhs.py`.  
3. **Start Training**:  
   ```bash
   python3 train_VNN_fusion_highQ.py
   ```
4. **Preprocessing** – On first execution, video frames will be preprocessed. This process **runs once** and speeds up subsequent training.  

---

## RKHS Volterra Model Architecture  

The **RKHS-based implementation** extends traditional Volterra filters by embedding inputs into a **high-dimensional Hilbert space**, allowing for:  
- **Nonlinear interactions** modeled with kernel functions.  
- **Regularization and better generalization**, reducing the risk of overfitting.  
- **Enhanced adaptability** to complex datasets.  

📌 The RKHS-based model is implemented in:  
```plaintext
networks/vnn_rkhs.py
```

---

## Project Structure  

This project follows a **modular structure** for clarity and scalability:  

```
volterra/
├── config/                  # Configuration files
├── data/                    # Raw datasets for training and testing
├── jobs/                    # Batch job scripts (e.g., SLURM for clusters)
├── logs/                    # Training and evaluation logs
├── models/                  # Saved models and training checkpoints
├── network/                 # Model architectures
│   ├── vnn_rgb_of_complex.py   # Original VNN architecture for complex inputs
│   ├── vnn_rkhs.py             # RKHS-based Volterra filter implementation
├── Inference.py              # Script for running inference on trained models
├── requirements.txt          # List of required Python dependencies
├── train_VNN_fusion_highQ.py # Main script for training VNN models
└── README.md                 # Project documentation
```

---

## Citation  

If you use this work, please cite the original authors and this extended implementation:  

```bibtex
@inproceedings{roheda2020conquering,
  title={Conquering the CNN Over-Parameterization Dilemma: A Volterra Filtering Approach for Action Recognition},
  author={Roheda, Siddharth and Krim, Hamid},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={34},
  number={07},
  pages={11948--11956},
  year={2020}
}

@article{roheda2019volterra,
  title={Volterra Neural Networks (VNNs)},
  author={Roheda, Siddharth and Krim, Hamid},
  journal={arXiv preprint arXiv:1910.09616},
  year={2019}
}
```

---

## Future Work  

🚀 **Pre-trained Models** – Provide pre-trained models for RKHS-based architectures.  
🔬 **Alternative Kernel Functions** – Experiment with different kernel functions for RKHS embeddings.  
🧠 **Hybrid VNN-CNN Models** – Explore hybrid architectures for large-scale datasets.  

---

## Contact  

For questions or feedback, feel free to reach out:  
- **Author**: Clément Leprêtre 
- **Institution**: CentraleSupélec  
- **Email**: clement.lepretre@student-cs.fr

---

## Acknowledgements  

This work builds upon the foundational contributions of **Siddharth Roheda and Hamid Krim**. Special thanks to **CentraleSupélec** for providing the research environment and resources.  

---
