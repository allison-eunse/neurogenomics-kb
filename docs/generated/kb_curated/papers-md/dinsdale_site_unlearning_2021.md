---
title: "Deep learning-based unlearning of dataset bias for MRI harmonisation and confound removal"
authors: "Natalie K. Dinsdale, Seong Jae Hwang, Stephen M. Smith, Christian F. Beckmann, Amalio Telenti, Thomas E. Nichols, Mark Jenkinson"
year: 2021
venue: "NeuroImage"
---

# Deep learning-based unlearning of dataset bias for MRI harmonisation and confound removal

**Authors:** Natalie K. Dinsdale, Seong Jae Hwang, Stephen M. Smith, Christian F. Beckmann, Amalio Telenti, Thomas E. Nichols, Mark Jenkinson  
**Year:** 2021  
**Venue:** NeuroImage^[https://www.sciencedirect.com/science/article/pii/S1053811920311745](https://www.sciencedirect.com/science/article/pii/S1053811920311745)

---

## 1. Classification

- **Domain Category:**  
  - MRI harmonization / confound removal. The paper introduces an adversarial “unlearning” framework to remove dataset (site) and other confounds from deep neural network representations while preserving task-relevant signal.

- **FM Usage Type:**  
  - Method / representation regularization. The approach is a training strategy that can be applied to CNNs or other encoders used in downstream prediction tasks; it is not a foundation model itself.

- **Key Modalities:**  
  - Structural MRI (e.g., T1-weighted) from multiple datasets/sites with varying acquisition protocols and subject characteristics.

---

## 2. Executive Summary

Dataset/site differences and other confounds (e.g., age, sex, scanner) can bias deep learning models trained on MRI data, leading to poor generalization and spurious associations. This paper proposes an **adversarial unlearning framework** that encourages the network’s internal representations to be **invariant** to specified nuisance variables (dataset, scanner, confounds) while retaining information useful for the main prediction task.

The method uses a standard CNN backbone for the main task (e.g., age prediction) and one or more adversarial branches that try to predict nuisance variables (e.g., dataset/site). During training, gradients from the adversarial branches are **reversed** (via a gradient reversal layer), so the shared feature extractor is encouraged to **remove** information that allows accurate nuisance prediction (“unlearning” the dataset bias). Experiments on multi-site MRI datasets show that this approach reduces site-related differences in learned features, improves cross-dataset generalization, and allows more accurate estimation of biological effects (e.g., age) independent of site. The framework is flexible: it can unlearn multiple confounds jointly and can be combined with existing architectures and tasks.

---

## 3. Problem Setup and Motivation

- **Scientific / practical problem:**  
  - Remove non-biological biases (dataset/site, acquisition protocol, confounds) from MRI-based deep learning models so that predictions reflect underlying biology rather than scanner/site artifacts.

- **Why this is hard:**  
  - **Entangled representations:** Standard CNNs naturally encode site and protocol information along with anatomical features.  
  - **Site imbalance:** Some sites/datasets may dominate training data, causing models to rely on dataset-specific cues.  
  - **Confound correlations:** Biological variables of interest (e.g., age, diagnosis) may be correlated with site or scanner, making it difficult to separate their effects.  
  - **Classical harmonization limits:** Intensity-based or ComBat-style methods operate at the image or feature level and may not fully remove deep feature-level confounds.

---

## 4. Data and Modalities

- **Datasets used:**  
  - Multiple structural MRI datasets from different sites/scanners, with variation in demographics and acquisition. (See paper for exact cohorts and sample sizes.)

- **Modalities:**  
  - Structural MRI (T1-weighted) volumes; method is agnostic to specific 3D CNN architecture.

- **Preprocessing / representation:**  
  - Standard MRI preprocessing: skull stripping, normalization, and resampling to a common space before CNN ingestion.  
  - CNN extracts intermediate feature maps that are shared between main and adversarial branches.

---

## 5. Model / Method

- **Model Type:**  
  - CNN-based predictor with adversarial branches for nuisance prediction and gradient reversal for unlearning.

- **Key components and innovations:**
  - **Gradient reversal layer (GRL):**  
    - During forward pass, GRL acts as identity; during backward pass, it multiplies gradients by \(-\lambda\), encouraging the shared encoder to make nuisance prediction difficult.  
  - **Multi-confound unlearning:**  
    - Multiple adversarial branches can target different nuisances (dataset, scanner, sex, etc.) simultaneously.  
  - **Joint optimization:**  
    - Loss = main task loss (e.g., age MAE) **minus** weighted nuisance prediction losses, balancing performance and invariance.

- **Training setup:**  
  - Mini-batch SGD with combined loss.  
  - Careful tuning of the GRL scaling parameter \(\lambda\) to avoid under- or over-unlearning.

---

## 6. Multimodal / Integration Aspects

- **Not multimodal:**  
  - Operates solely on MRI images, but its representations can be used as inputs to multimodal models.

- **Integration relevance:**  
  - Provides a recipe for **representation-level harmonization** that can be applied to encoders used in gene–brain or brain–behavior integration:  
    - Learn brain embeddings that are invariant to site while still predictive of age, diagnosis, or other targets.  
    - Reduce spurious correlations between site and downstream genetics or behavioral variables.

---

## 7. Experiments and Results

### Main findings

- Models trained with unlearning show **reduced dataset bias** in feature representations, as measured by adversarial classifier accuracy and visualization of embeddings.  
- **Cross-dataset generalization** improves: models trained on one set of datasets perform better on held-out datasets when unlearning is applied.  
- **Biological signal retention:** Age and disease-related effects remain or improve in models with unlearning, suggesting that useful signal is preserved while nuisance signal is attenuated.

### Comparisons

- Outperforms or complements traditional harmonization approaches by operating directly at the representation level within deep networks.  
- Demonstrates that adversarial unlearning is a viable alternative to purely image-level or feature-level statistical harmonization.

---

## 8. Strengths and Limitations

### Strengths

- **Flexible and model-agnostic:**  
  - Can be attached to many CNN (or other encoder) architectures with minimal changes.  
  - Handles multiple confounds concurrently.

- **Directly targets representation bias:**  
  - Encourages invariance precisely where it matters—inside the learned feature space.

- **Improves generalization:**  
  - Better cross-dataset performance and more reliable effect estimates.

### Limitations

- **Hyperparameter sensitivity:**  
  - Performance depends on the choice of GRL scaling and loss weights.  
  - Over-aggressive unlearning can remove task-relevant information.

- **Confound specification:**  
  - Requires explicit labels for nuisances (e.g., site IDs, scanner types).  
  - Unobserved confounds may still leak into representations.

---

## 9. Context and Broader Impact

- **Relation to domain adaptation and fairness:**  
  - Connects to adversarial domain adaptation and fairness literature where GRL-based unlearning is used to remove protected attributes (e.g., gender, race) from embeddings.  
  - Provides a concrete instantiation for neuroimaging where site/scanner is the “protected” attribute.

- **Impact on large-scale neuroimaging consortia:**  
  - Offers tools for harmonizing representations across datasets without discarding data or overly restricting models.  
  - Supports multi-cohort analyses and meta-analyses with improved robustness to site confounds.

---

## 10. Key Takeaways

1. **Adversarial unlearning can remove dataset/site bias from MRI-based deep models** while retaining task-relevant information.  
2. **Gradient reversal layers enable simple, end-to-end implementation** of unlearning in existing encoders.  
3. **Representation-level harmonization complements image-level methods** like ComBat and MURD, offering additional control over confounds.  
4. **Multi-confound unlearning is feasible**, allowing simultaneous control of site, scanner, and other nuisance variables.  
5. **Method is directly relevant to integration pipelines** that rely on site-robust brain embeddings for gene–brain–behavior analysis.



