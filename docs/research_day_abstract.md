# Research Day 2026 - Abstract Submission

**Title:** Beyond the Biopsy: Transforming Liver Fibrosis Staging with Advanced Vision Transformers and Efficient Neural Networks

**Abstract:**

Liver fibrosis is a silent but progressive condition that can lead to irreversible cirrhosis and liver failure if left unchecked. The current gold standard for diagnosis—histopathological assessment of liver biopsies—is invasive, subjective, and prone to significant inter-observer variability among pathologists. This research proposes a robust, automated diagnostic framework designed to standardize fibrosis staging and assist clinicians in making earlier, more accurate diagnoses.
Our study introduces a novel ensemble approach that synergizes the structural grasp of Convolutional Neural Networks (CNNs) with the global context capabilities of Vision Transformers (ViT). We rigorously evaluated three distinct architectures: the classic ResNet50, the highly optimized EfficientNet-V2, and the cutting-edge ViT-B/16. Unlike traditional methods that struggle with the subtle visual nuances between intermediate fibrosis stages (F1-F3), our pipeline applies specific preprocessing techniques, such as Contrast Limited Adaptive Histogram Equalization (CLAHE), to enhance tissue feature visibility before model training.
The results represent a significant leap forward in automated histopathology. While the standard ResNet50 model achieved a respectable 91.30% accuracy, our optimized EfficientNet-V2 model reached 96.60%, demonstrating exceptional efficiency. Most notably, the Vision Transformer model outperformed both, achieving a remarkable 97.47% diagnostic accuracy. This performance proves that attention-based models can effectively discern complex fibrosis patterns that even experienced eyes might miss.
By mitigating human error and providing a consistent, high-precision second opinion, this tool holds the potential to revolutionize pathology workflows. It offers a pathway to faster, reliable staging, ultimately facilitating timely therapeutic interventions and better patient outcomes in chronic liver disease management.

**Keywords:** Liver Fibrosis, Vision Transformers, EfficientNet, Deep Learning, Digital Pathology
