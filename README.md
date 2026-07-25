# King County Housing Segmentation

**Data Driven Modelling – TH Köln**

## Project Overview

This project analyzes the King County housing market using clustering methods.

The goal is to identify meaningful property segments and derive possible marketing strategies for each segment.

## Dataset

- King County House Sales Dataset (2014–2015)
- 21,613 house sales
- No missing values after preprocessing

**Note:**  
The dataset is not included in this repository.

## Data Preparation

The following preprocessing steps were performed:

- Correction of the known bedroom outlier (33 → 3)
- Creation of `house_age`
- Creation of `house_age_sq`
- Creation of `renovated`
- Log transformation of lot size
- Feature standardization using `StandardScaler`

The house price was **not** used as a clustering feature. It was only used afterwards for business interpretation.

## Clustering Methods

Three clustering algorithms were evaluated:

- K-Means
- Agglomerative Clustering (Ward linkage)
- DBSCAN

For the final business analysis, **Agglomerative Clustering (k = 4)** was selected because it produced the most interpretable market segments.

## Final Segments

- Newer Family Houses
- Older Standard Houses
- Waterfront Luxury Houses
- Renovated Older Houses

## Repository Structure

```
DDM-KingCounty
│
├── DDM_KingCounty_Final_Complete.ipynb
├── ddm_kingcounty_final.py
├── requirements.txt
│
├── presentation/
│   └── Final_Presentation.pptx
│
└── outputs/
    ├── agg_clustering_map_powerpoint.png
    ├── agg_price_boxplot_powerpoint.png
    ├── final_model_comparison.csv
    └── selected_clustering_profile.csv
```

## Authors

- Yassine Hani
- Samuel Habte

TH Köln – Data Driven Modelling
