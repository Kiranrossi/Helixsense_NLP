---
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: 301560800487 supply of hand dryer - hand dryer - eh270n - euronics hand dryer
    eh270n -  make - euronics
- text: camera cctv -ds-2ce76d0t-itpfs-hd dome camera 2mp with switch mode button-hikvis
- text: supply of wheel chair  wheel chair folding type dim   42  l x 22  w x 36  h
    with 17  seat
- text: eye wash portable unit  ewp 07   make - euronics/equivalent  size - 540mm
    255mm
- text: pinnacle po 28151 shoe shine machine t7 qty01
metrics:
- accuracy
pipeline_tag: text-classification
library_name: setfit
inference: true
base_model: sentence-transformers/paraphrase-mpnet-base-v2
---

# SetFit with sentence-transformers/paraphrase-mpnet-base-v2

This is a [SetFit](https://github.com/huggingface/setfit) model that can be used for Text Classification. This SetFit model uses [sentence-transformers/paraphrase-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-mpnet-base-v2) as the Sentence Transformer embedding model. A [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance is used for classification.

The model has been trained using an efficient few-shot learning technique that involves:

1. Fine-tuning a [Sentence Transformer](https://www.sbert.net) with contrastive learning.
2. Training a classification head with features from the fine-tuned Sentence Transformer.

## Model Details

### Model Description
- **Model Type:** SetFit
- **Sentence Transformer body:** [sentence-transformers/paraphrase-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-mpnet-base-v2)
- **Classification head:** a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance
- **Maximum Sequence Length:** 512 tokens
- **Number of Classes:** 3 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

### Model Labels
| Label     | Examples                                                                                                                                                                                                                                                                             |
|:----------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Equipment | <ul><li>'dr aqua po 25956 ro water centralization water coolert2'</li><li>'giri enterprises po 27729 chair evacuation t7 lg floorqty03'</li><li>'euronics po 28319 sanitary pad dispenser qty 01'</li></ul>                                                                          |
| Services  | <ul><li>'supply  installation  testing and commissioning of hi wall fan coil unit 2tr wit'</li><li>'sitc of converting ph-1 analog cctv camera into ip camera  company-siemens'</li><li>'services professional services   consultancy architects services design at ameni'</li></ul> |
| Material  | <ul><li>'insullation on 32mm dia pipe / valves with 19mm nitril foam closed cell elastome'</li><li>'aaztec po 28225 digital directory led ki'</li><li>'supply   fixing of ms conduit pipe'</li></ul>                                                                                 |

## Uses

### Direct Use for Inference

First install the SetFit library:

```bash
pip install setfit
```

Then you can load this model and run inference.

```python
from setfit import SetFitModel

# Download from the 🤗 Hub
model = SetFitModel.from_pretrained("setfit_model_id")
# Run inference
preds = model("pinnacle po 28151 shoe shine machine t7 qty01")
```

<!--
### Downstream Use

*List how someone could finetune this model on their own dataset.*
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Set Metrics
| Training set | Min | Median  | Max |
|:-------------|:----|:--------|:----|
| Word count   | 1   | 11.5446 | 48  |

| Label     | Training Sample Count |
|:----------|:----------------------|
| Services  | 49                    |
| Equipment | 113                   |
| Material  | 51                    |

### Training Hyperparameters
- batch_size: (8, 8)
- num_epochs: (1, 1)
- max_steps: -1
- sampling_strategy: oversampling
- num_iterations: 10
- body_learning_rate: (1e-05, 1e-05)
- head_learning_rate: 0.01
- loss: CosineSimilarityLoss
- distance_metric: cosine_distance
- margin: 0.25
- end_to_end: False
- use_amp: False
- warmup_proportion: 0.1
- l2_weight: 0.01
- max_length: 128
- seed: 42
- eval_max_steps: -1
- load_best_model_at_end: False

### Training Results
| Epoch  | Step | Training Loss | Validation Loss |
|:------:|:----:|:-------------:|:---------------:|
| 0.0019 | 1    | 0.4212        | -               |
| 0.0938 | 50   | 0.2472        | -               |
| 0.1876 | 100  | 0.2047        | -               |
| 0.2814 | 150  | 0.1576        | -               |
| 0.3752 | 200  | 0.1133        | -               |
| 0.4690 | 250  | 0.0918        | -               |
| 0.5629 | 300  | 0.0711        | -               |
| 0.6567 | 350  | 0.0498        | -               |
| 0.7505 | 400  | 0.0359        | -               |
| 0.8443 | 450  | 0.0301        | -               |
| 0.9381 | 500  | 0.0147        | -               |

### Framework Versions
- Python: 3.13.7
- SetFit: 1.1.3
- Sentence Transformers: 5.2.0
- Transformers: 4.57.3
- PyTorch: 2.8.0
- Datasets: 4.4.1
- Tokenizers: 0.22.1

## Citation

### BibTeX
```bibtex
@article{https://doi.org/10.48550/arxiv.2209.11055,
    doi = {10.48550/ARXIV.2209.11055},
    url = {https://arxiv.org/abs/2209.11055},
    author = {Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke and Korat, Daniel and Wasserblat, Moshe and Pereg, Oren},
    keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Efficient Few-Shot Learning Without Prompts},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->