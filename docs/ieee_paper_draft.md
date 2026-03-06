# Automated Liver Fibrosis Staging using an Ensemble of CNN and Vision Transformer Architectures

*R. Pandian*
*Department of Computer Science*
*University*
*Email: rpandian@example.com*

---

**_Abstract_—Liver fibrosis is a major health concern, and its accurate staging is critical for effective treatment planning. The METAVIR scoring system (F0 to F4) is commonly used, but manual pathological assessment is time-consuming, subjective, and prone to inter-observer variability. In this work, we propose a robust automated system for liver fibrosis staging using a deep learning ensemble approach. Our method combines the strengths of various modern architectures, including ResNet50, ConvNeXt Tiny, and DeiT-Small. To address inherent class imbalance in medical datasets, we employ a WeightedRandomSampler during training. We evaluated our models on a test set of 1,265 samples. While the individual ConvNeXt model achieved an impressive accuracy of 98.42%, our proposed ensemble pipeline mitigates severe misclassifications and achieves a near-perfect Cohen’s Kappa score of 0.9938. The results demonstrate the potential of this automated system to serve as a reliable second opinion in clinical settings, promising enhanced diagnostic efficiency and consistency.**

**_Keywords_—Liver Fibrosis, METAVIR, Deep Learning, Ensemble, CNN, Vision Transformer, ConvNeXt, DeiT.**

## I. INTRODUCTION

Liver fibrosis represents the scarring process leading to cirrhosis, a major global health issue. The precise determination of the fibrosis stage, typically assessed using the METAVIR scoring system ranging from F0 (no fibrosis) to F4 (cirrhosis), plays a pivotal role in formulating clinical decisions, treatment regimens, and prognostic predictions.

Traditionally, liver fibrosis staging hinges on the manual microscopic examination of biopsy samples by expert pathologists. Notwithstanding its status as the gold standard, manual staging is fraught with challenges. Primarily, it is an exhaustive, time-intensive process subject to considerable inter-observer and intra-observer variability. The inherently subjective nature of visual assessment, coupled with the fatigue of evaluating numerous slides, amplifies the risk of diagnostic inconsistencies.

To mitigate these limitations, we present the Automated Liver Staging (ALS) system. Our solution leverages state-of-the-art Deep Learning (DL) architectures to deliver a rapid, objective, and reproducible evaluation pipeline. This paper details the comparative assessment and deployment of Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs), culminating in a robust ensemble model.

## II. RELATED WORK

Early efforts in computational pathology for liver fibrosis often relied on hand-crafted features extracted via traditional image processing techniques and classified using algorithms like Support Vector Machines (SVMs). While these laid the groundwork, they struggled with the heterogeneous appearance of fibrosis across different samples and staining protocols.

The advent of deep learning, particularly CNNs, revolutionized the field. Models such as ResNet and Inception have demonstrated significant success in medical image classification tasks by automatically extracting hierarchical feature representations. Recent studies have extended these architectures to histological image analysis with promising accuracy.

However, CNNs exhibit a localized receptive field, which can limit their capacity to capture long-range contextual dependencies crucial for whole-slide comprehending. Vision Transformers (ViTs) have recently emerged to address this constraint by applying self-attention mechanisms across image patches. In this paper, we bridge the gap by bringing modern CNNs like ConvNeXt—which adopts several design choices from ViTs—and DeiT (Data-efficient Image Transformers) into a unified ensemble framework for optimal liver staging.

## III. METHODOLOGY

### A. Dataset and Preprocessing

The dataset encompasses diverse histological samples meticulously annotated by experienced pathologists according to the METAVIR staging system (F0-F4). For robust evaluation, a dedicated test set of 1,265 samples was isolated.

A critical challenge in medical datasets is class imbalance, particularly the over-representation of lower severity stages. To counteract this bias and prevent the model from leaning towards the majority classes, a `WeightedRandomSampler` was integrated into the PyTorch DataLoader, ensuring balanced mini-batch sampling during the training phase. Standard data augmentation techniques were further employed to enhance generalization.

### B. Network Architectures

We conducted a comparative study evaluating three distinct deep learning architectures:
1) **ResNet50:** Serving as our robust baseline CNN, renowned for its residual learning framework that facilitates the training of deep networks without vanishing gradients.
2) **ConvNeXt Tiny:** A modernized CNN architecture that integrates design concepts from Vision Transformers (such as larger kernel sizes and altered activation functions), achieving competitive performance with ViTs while retaining the inductive biases of CNNs.
3) **DeiT-Small:** A Vision Transformer optimized for data-efficient training. DeiT relies entirely on attention mechanisms to capture global context across image patches, offering an alternative paradigm to spatial convolutions.

### C. Ensemble Strategy

To harness the complementary strengths of these disparate architectures, we designed an ensemble pipeline. The `run_ensemble_pathologist.py` module aggregates the predictive probabilities derived from each individual model (ResNet50, ConvNeXt, DeiT). By fusing these predictions, the ensemble aims to smooth out the variance of individual models and significantly reduce the likelihood of severe, multi-stage misclassifications.

## IV. RESULTS AND DISCUSSION

The experimental evaluation highlights the efficacy of the proposed models, particularly the ConvNeXt model and the Ensemble pipeline. Performance was quantified using standard Accuracy and Cohen's Kappa, the latter being critical for ordinal classification tasks like staging, as it heavily penalizes drastic misclassifications (e.g., predicting F0 when the true stage is F4).

### A. Performance Evaluation

Our empirical results, illustrated below, confirm the robustness of the system. While the ConvNeXt Tiny model achieved the highest raw accuracy, the Ensemble model exhibited superior agreement with ground truth annotations.

_Evaluation Metrics:_
- **Ensemble (All Models):** Accuracy 98.26%, Cohen's Kappa 0.9938
- **ConvNeXt Tiny:** Accuracy 98.42%, Cohen's Kappa 0.9793
- **ResNet50:** Accuracy 91.30%, Cohen's Kappa 0.8900
- **DeiT-Small:** Accuracy 85.53%, Cohen's Kappa 0.8200

### B. Discussion

The ConvNeXt model demonstrates exceptional individual performance, marginally outperforming the ensemble by accuracy (98.42%). However, the Ensemble achieved the pinnacle Cohen's Kappa score of 0.9938. This metric indicates near-perfect agreement, underscoring the ensemble's capacity to minimize critical clinical errors. The relatively lower performance of DeiT-Small (85.53%) aligns with the acknowledged data-hungry nature of standard Vision Transformers, suggesting that while attention mechanisms are powerful, convolutional inductive biases (as seen in ConvNeXt) remain highly effective on mid-sized medical datasets.

_(Figure 1: Placeholder for Confusion Matrix)_

## V. CONCLUSION

We presented an Automated Liver Staging system leveraging an ensemble of ResNet, ConvNeXt, and DeiT architectures. The integration of a WeightedRandomSampler successfully mitigated inherent dataset imbalances. Our results validate the superiority of the ensemble approach with a near-perfect Cohen's Kappa of 0.9938, demonstrating its viability as a reliable computational adjunct for pathologists. Future work will focus on expanding the dataset across multiple centers to assess broader generalization and further refining Vision Transformer models for histological applications.
