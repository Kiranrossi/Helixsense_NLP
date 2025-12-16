---
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: dr aqua po 25956 ro water centralization water coolert3
- text: sitc of camera licence for server  exacqvision
- text: dismantling existing aircool unit with shifting as per instruction
- text: euronics po 28319 sanitary pad dispenser qty 01
- text: global engineering 28216 aqi monitoring
metrics:
- accuracy
pipeline_tag: text-classification
library_name: setfit
inference: true
base_model: sentence-transformers/all-MiniLM-L6-v2
---

# SetFit with sentence-transformers/all-MiniLM-L6-v2

This is a [SetFit](https://github.com/huggingface/setfit) model that can be used for Text Classification. This SetFit model uses [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) as the Sentence Transformer embedding model. A [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance is used for classification.

The model has been trained using an efficient few-shot learning technique that involves:

1. Fine-tuning a [Sentence Transformer](https://www.sbert.net) with contrastive learning.
2. Training a classification head with features from the fine-tuned Sentence Transformer.

## Model Details

### Model Description
- **Model Type:** SetFit
- **Sentence Transformer body:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Classification head:** a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance
- **Maximum Sequence Length:** 256 tokens
- **Number of Classes:** 3 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

### Model Labels
| Label     | Examples                                                                                                                                                                                                                       |
|:----------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Equipment | <ul><li>'eye wash portable unit  ewp 07   make - euronics equivalent  size - 540mm 255mm'</li><li>'mma design po 27347 chairs for food courtqty20'</li><li>'sitc ms elbow 2'</li></ul>                                         |
| Services  | <ul><li>'provision reclass of godrej may22'</li><li>'interior and mep work of tower-3 gf   ff atrium as per attach order'</li><li>'supply  installation  testing and commissioning of hi wall fan coil unit 2tr wit'</li></ul> |
| Material  | <ul><li>'supply installation of pipe pvc drain line pipe 1  dia size along with necessar'</li><li>'insullation on 1 5  dia pipe   valves with 19mm nitril foam closed cell elastome'</li><li>'sitc jumper wire'</li></ul>      |

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
preds = model("global engineering 28216 aqi monitoring")
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
| Word count   | 1   | 11.9484 | 59  |

| Label     | Training Sample Count |
|:----------|:----------------------|
| Services  | 71                    |
| Equipment | 101                   |
| Material  | 41                    |

### Training Hyperparameters
- batch_size: (32, 32)
- num_epochs: (4, 4)
- max_steps: -1
- sampling_strategy: unique
- num_iterations: 10
- body_learning_rate: (2e-05, 2e-05)
- head_learning_rate: 0.01
- loss: CosineSimilarityLoss
- distance_metric: cosine_distance
- margin: 0.25
- end_to_end: False
- use_amp: False
- warmup_proportion: 0.0
- l2_weight: 0.01
- max_length: 96
- seed: 42
- eval_max_steps: -1
- load_best_model_at_end: False

### Training Results
| Epoch  | Step | Training Loss | Validation Loss |
|:------:|:----:|:-------------:|:---------------:|
| 0.0075 | 1    | 0.3687        | -               |
| 0.3731 | 50   | 0.2294        | -               |
| 0.7463 | 100  | 0.1714        | -               |
| 1.1194 | 150  | 0.1126        | -               |
| 1.4925 | 200  | 0.0693        | -               |
| 1.8657 | 250  | 0.061         | -               |
| 2.2388 | 300  | 0.0412        | -               |
| 2.6119 | 350  | 0.0363        | -               |
| 2.9851 | 400  | 0.0331        | -               |
| 3.3582 | 450  | 0.0349        | -               |
| 3.7313 | 500  | 0.0272        | -               |

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