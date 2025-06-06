



<div align="right">
    <img src="25_0606_SymbolicXAI_HI__COLOUR_transparentBG.png" width="500" style="margin-top:-60px; margin-bottom:0;"/>
</div>

# Official repo of the paper: <br> *Towards symbolic XAI — explanation through human understandable logical relationships between features* 



## Installation instruction
When you want to install everything using pip, just write
```bash
pip install -r requirements.txt
```
to install the requirements. And to install this module as a package write
```bash
pip install -e .
```
which will call `setup.py`, where the `-e` flag will make sure changes in the repository will also affect the module. You can can call the module afterwards by `import symbxai`. 

## Reconstructing the results

| Result                             | Code                                                   |
|-------------------------------------|--------------------------------------------------------|
| **Figure 1 a)** NLP example        | `mult-order_query_qualitative_generation.ipynb`       |
| **Figure 1 b)** Vision example     | `symbXAI_vision_figures.ipynb`                        |
| **Figure 1 b)** Quantum chemistry  | `quantum_chemistry_MDA_Figure1.ipynb`                 |
| **Figure J.19**                    | `quantum_chemistry_analyze_MDA_trajectory.ipynb`      |
| **Figure 2**, **Figure 3 a)**       | `mult-order_query_qualitative_generation.ipynb`       |
| **Figure 3 b)**                     | `symb_Vis_multi_order_fig2.ipynb`                     |
| **Figure 6**                        | `nlp_sst_exp.ipynb`                                                  |
| **Figure 7**                        | `Automatization process on contrastive conjunctions.ipynb` |
| **Table 2**, **Figure G.13, G.14**  |  `scripts/perform_perturbation.py` and `perturbation_analysis.ipynb`|
| **Figure 8**                        | -                                                 |
| **Figure 9**, **Table 4**           | `vision_query_search_exp.ipynb`                                                    |
| **Figure 10**, **H.16, H.17**       | `mutag_explainations.ipynb`                          |
| **Figure D.11, D.12**               | `nlp_movie_reviews_exp.ipynb`                                                   |
| **Figure H.15, H.15**               | NOT CLEAN YET                                                   |


## Other dependencies:
### Datasets
Some precomputed data can be found in  `notebooks/data` folder.

For the Facial Expression Recognition task one can download the data under https://www.kaggle.com/datasets/msambare/fer2013


### Citation
If you want to cite us please use this BibTex entry
```LaTex
@article{schnake2025symbxai,
    title = {Towards symbolic XAI — explanation through human understandable logical relationships between features},
    journal = {Information Fusion},
    volume = {118},
    pages = {102923},
    year = {2025},
    issn = {1566-2535},
    doi = {https://doi.org/10.1016/j.inffus.2024.102923},
    author = {Thomas Schnake and Farnoush {Rezaei Jafari} and Jonas Lederer and Ping Xiong and Shinichi Nakajima and Stefan Gugler and Grégoire Montavon and Klaus-Robert Müller},
    keywords = {Explainable AI, Concept relevance, Higher-order explanation, Transformers, Graph neural networks, Symbolic AI},
}
```
