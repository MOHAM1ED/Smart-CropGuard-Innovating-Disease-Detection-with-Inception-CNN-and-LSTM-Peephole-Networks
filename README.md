# Smart-CropGuard-Innovating-Disease-Detection-with-Inception-CNN-and-LSTM-Peephole-Networks
This study proposes a deep learning model combining Inception ResNet V2 and LSTM with peephole connections for plant disease detection. It achieves high accuracy and efficiency in disease classification, enhancing agricultural sustainability. For full details, visit [IEEE Xplore](https://ieeexplore.ieee.org/document/10620471).

Here’s an updated version of the README with the conclusion and references you’ve provided, integrated into the structure:

---

# Smart CropGuard: Disease Detection with Inception CNN and LSTM Peephole Networks

This project integrates advanced deep learning techniques, combining **Inception Convolutional Neural Networks (CNN)** and **Long Short-Term Memory (LSTM)** networks with **peephole connections** for automated plant disease detection. By leveraging state-of-the-art architectures, it enhances agricultural sustainability through accurate, real-time disease classification.

## Table of Contents
- [Introduction](#introduction)
- [Methodology](#methodology)
- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)
- [Conclusion](#conclusion)
- [License](#license)
- [References](#references)

## Introduction

Detecting plant leaf diseases is critical for improving agricultural productivity and sustainability. Traditional manual inspection methods are time-consuming and prone to errors. This project aims to address these challenges by utilizing deep learning models to automate and enhance disease detection.

Key technologies used:
- **Inception CNN**: A deep learning model that extracts detailed image features using multiple convolutional layers.
- **LSTM with Peephole Connections**: Designed for sequential data, this model captures long-term dependencies, enhancing temporal analysis.

## Methodology

The model is trained using the **Plant Village dataset**, which contains 38 crop-disease pairs. The architecture integrates:
1. **Inception ResNet V2**: For multi-scale feature extraction.
2. **LSTM with Peephole Connections**: For capturing temporal dynamics in sequential image data.

### Key Components:
- **LSTM with Peephole Connections**: Enhances LSTM's ability to capture long-term dependencies by allowing gates to access the internal cell state. This is crucial for sequential image data and temporal dynamics.
  
- **Inception ResNet V2**: Incorporates residual connections in the Inception modules, facilitating deeper networks and mitigating issues like vanishing gradients.

### Training Process:
- Utilized **Google Colab** with **T4 GPU** for faster training, enabling efficient processing of large datasets like Plant Village.

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/username/SmartCropGuard.git
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Plant Village dataset** and place it in the `data` folder.

## Usage

### Training the Model:
```bash
python train_model.py
```

### Making Predictions:
```bash
python predict.py --image <image_path>
```

## Results

The model achieved:
- **Accuracy**: 92.39% on the test set.
- **Precision**: 93.73%
- **Recall**: 90.60%
- **AUC**: 99.59%

These results demonstrate the model’s effectiveness in classifying plant diseases accurately.

## Conclusion

This study represents a significant advancement in automated plant leaf disease detection, utilizing **Inception ResNet V2** and **LSTM networks with peephole connections**. The integration of these advanced techniques has resulted in a robust framework that enhances disease detection accuracy and efficiency, addressing critical agricultural challenges. The model demonstrated substantial progress in performance over successive training epochs, with impressive results on both validation and test datasets.

Evaluation metrics such as accuracy, precision, recall, and AUC confirm the model's ability to effectively identify and classify plant diseases. The incorporation of peephole connections in the LSTM architecture contributed to its capability to capture temporal patterns, improving its sequential data processing. The model's ability to distinguish between different disease classes highlights its potential for real-world application, where timely disease identification is crucial for crop management and food security.

Looking forward, ongoing refinements, further exploration of additional datasets, and deployment in various agricultural scenarios will continue to drive innovation in plant disease detection, fostering more sustainable agricultural practices.

## License

Distributed under the MIT License. See `LICENSE` for more information.

Here’s the updated **References** section with the link to your paper:

---

## References

1. Dathan, R. A., & Shanmugapriya, S. Yield forecast of soyabean crop using Peephole LSTM Framework. *Lecture Notes in Networks and Systems*, 261–270. [DOI](https://doi.org/10.1007/978-981-19-3148-2_22)
2. Liu, B., Zhang, Y., He, D., & Li, Y. Identification of apple leaf diseases based on deep convolutional neural networks. *Symmetry*, 10(1), 11. [DOI](https://doi.org/10.3390/sym10010011)
3. Jabir, B., et al. A strategic Analytics using convolutional neural networks for weed identification in sugar beet fields. *AGRIS On-line Papers in Economics and Informatics*, 13(1), 49–57. [DOI](https://doi.org/10.7160/aol.2021.130104)
4. Fang, Y., Ramasamy, R. P., & Rains, G. C. “Current and prospective methods for plant disease detection.” *Biosensors*, 5, 537–561. [DOI](10.3390/bios5030537)
5. Durmuş, H., Güneş, E. O., & Kırcı, M. Disease detection on the leaves of tomato plants using deep learning. *6th International Conference on Agro-Geoinformatics*, 1–5. [DOI](10.1109/Agro-Geoinformatics.2017.8047016)
6. Sutskever, I., Vinyals, O., & Le, Q. V. “Sequence to sequence learning with neural networks,” in *Advances in Neural Information Processing Systems*, 3104–3112.
7. LeCun, Y., Bengio, Y. & Hinton, G. Deep Learning. *Nature*, 521, 436-444. [DOI](https://doi.org/10.1038/nature14539)
8. Lecun, Y., Bottou, L., Bengio, Y., & Haffner, P. Gradient-Based Learning Applied to Document Recognition. *Proceedings of the IEEE*, 86, 2278-2324. [DOI](https://doi.org/10.1109/5.726791)
9. Disease classification in maize crop using bag of features and multiclass support vector machine. *IEEE Xplore*. [Link](https://ieeexplore.ieee.org/abstract/document/8398993)
10. Lee, S. H., Goëau, H., Bonnet, P., & Joly, A. Attention-Based Recurrent neural network for plant disease classification. *Frontiers in Plant Science*, 11. [DOI](https://doi.org/10.3389/fpls.2020.601250)
11. Faye, M., Chen, B., & Sada, K. A. Plant Disease Detection with Deep Learning and Feature Extraction Using Plant Village. *Journal of Computer and Communications*, 08(06), 10–22. [DOI](https://doi.org/10.4236/jcc.2020.86002)
12. [My Paper](https://ieeexplore.ieee.org/document/10620471)
13. 
---
