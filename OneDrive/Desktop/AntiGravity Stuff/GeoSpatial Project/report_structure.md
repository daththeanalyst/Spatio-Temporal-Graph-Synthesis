# Report Structure: Predictive Site Selection for Specialty Coffee in Camden

**Target Length**: ~2,000 words (excluding references, figures, and tables)
**Format**: PDF via LaTeX or Jupyter-to-PDF export
**Style**: Academic — third person, past tense for methods/results, present tense for established theory

---

## Abstract (~150 words)

**Purpose**: Summarise the entire project in one self-contained paragraph.

**Content**:
- **Problem**: Identifying optimal locations for a new specialty coffee shop in the London Borough of Camden using multi-modal geospatial data.
- **Method**: Constructed a binary classification model on H3 hexagonal spatial units (Resolution 9, ~174m). Engineered 12 features from three modalities: LandScan population rasters (footfall proxy), ONS Census 2021 demographics (demand indicators), and NetworkX graph centrality metrics (urban connectivity). Compared Logistic Regression, Random Forest, and XGBoost using Spatial Block Cross-Validation to prevent spatial leakage. Tuned the best model via GridSearchCV.
- **Key Finding**: State the winning model, its AUC, and the number of False Positive hexagons identified as site recommendations.
- **Business Implication**: One sentence connecting FP hexes to Burt's Structural Hole Theory.

**No figures/tables in the abstract.**

---

## 1. Introduction (~250 words)

**Narrative arc**: Why this problem matters → What gap exists in the literature → What we do about it.

### Content:
- **Opening hook**: The rise of the "15-Minute City" concept (Moreno et al., 2021) and its implications for hyper-local retail placement. Frame the business context: specialty coffee as a growing market segment sensitive to demographics and urban connectivity.
- **Literature context**: Burt's Structural Hole Theory (1992) — originally a sociological concept about brokerage positions in social networks. We adapt it to spatial retail networks: a "structural hole" is a location with high demand signals but no existing supply.
- **Research gap**: Traditional site selection relies on heuristic scoring or expensive footfall surveys. There is limited work applying supervised ML to H3 hexagonal grids for micro-scale retail siting.
- **Research question**: *"Can a binary classification model, trained on geospatial features derived from population rasters, census demographics, and graph centrality metrics, identify underserved locations (structural holes) for specialty coffee in Camden?"*
- **Contribution**: (1) A reproducible ML pipeline on open data. (2) A novel application of Spatial Block CV using H3 hierarchy. (3) The False Positive interpretation as a business recommendation engine.

### Figures/Tables:
| ID | Type | Description | Source Cell |
|----|------|-------------|-------------|
| Fig. 1 | Map | Camden H3 hexagonal grid (Res 9) with borough boundary overlay | Notebook 02, cell-4 output |

---

## 2. Data & Feature Engineering (~350 words)

**Narrative arc**: What data we have → How we combined it → What features we built.

### Content:

#### 2.1 Data Sources
- **LandScan Global Population** (Oak Ridge National Laboratory, 2023): ~1km raster resampled to H3 Res-9 hexagons via zonal statistics (sum aggregation). Provides the `population` feature as a footfall proxy.
- **ONS Census 2021** (via EDINA Digimap): Three datasets at Output Area (OA) level — Economic Activity, Age Structure, Qualifications. 846 OAs covering Camden. Joined to hexagons via `sjoin_nearest` with population-weighted means. Key features: `employed_total_perc`, `age_16_to_34_perc`, `level4_perc` (degree-level education).
- **OpenStreetMap** (via OSMnx): Points of Interest classified into Competitor (cafe, coffee_shop), Synergy (gym, university, office, library), and Anchor (transit station) roles. Aggregated as counts per hexagon.
- **H3 Adjacency Graph** (via NetworkX): Hexagons as nodes, edges between adjacent hexes. Four centrality metrics computed: degree, betweenness, closeness, clustering coefficient.

#### 2.2 Feature Matrix
Present the complete feature dictionary.

#### 2.3 Target Variable
- Binary: `has_coffee_shop = 1` if hex contains >= 1 cafe/coffee_shop, else 0.
- **Leakage safeguard**: `n_competitors` is excluded from features because it directly encodes the target. `n_synergy` and `n_anchors` are retained as they are causally upstream (synergy nodes attract coffee shops, not vice versa).

### Figures/Tables:
| ID | Type | Description | Source Cell |
|----|------|-------------|-------------|
| Table 1 | Feature Dictionary | 12 features: name, source modality, data type, expected direction (+/-) with target | Manual |
| Fig. 2 | Heatmap | Pearson correlation matrix of all 12 features + target | Notebook ML, Section 6a |
| Fig. 3 | Box plots | Distribution of `population` and `level4_perc` by target class (0 vs 1) | Notebook ML, Section 6a |

---

## 3. Methodology (~400 words)

**Narrative arc**: Why binary classification → Why spatial CV is critical → What models we compared → How we tuned.

### Content:

#### 3.1 Problem Formulation
- Framed as binary classification: predict whether a hexagon "should" contain a coffee shop based on its geospatial features.
- The commercially valuable output is not accuracy per se, but the **False Positives** — hexagons the model predicts as 1 (suitable) that are currently 0 (no coffee shop). These represent structural holes.

#### 3.2 Spatial Cross-Validation
- **Motivation**: Tobler's First Law of Geography (1970): "Everything is related to everything else, but near things are more related than distant things." Random k-fold CV violates spatial independence — adjacent hexes with shared features can leak into both train and test sets, inflating AUC by 5–15%.
- **Implementation**: Custom `SpatialKFold` class. H3 Res-9 hexes are grouped by their Res-5 parent cell (~10km blocks). All hexes sharing a parent are assigned to the same fold. 5 folds, stratified by spatial block.
- **Formula**: $\text{fold}(H) = \text{h3\_cell\_to\_parent}(H, \text{res}=5) \bmod k$

#### 3.3 Model Comparison
Three models chosen to span the complexity spectrum:
1. **Logistic Regression** (baseline): Linear decision boundary. StandardScaler applied. `class_weight='balanced'`.
2. **Random Forest**: Non-linear, ensemble of decision trees. `class_weight='balanced'`, 200 estimators.
3. **XGBoost**: Gradient boosted trees. `scale_pos_weight` set to the class imbalance ratio.

#### 3.4 Hyperparameter Tuning
GridSearchCV on the best-performing model with the spatial CV splitter. Tuned: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`.

#### 3.5 Class Imbalance Handling
- Used built-in class weighting (not SMOTE). Rationale: SMOTE generates synthetic samples in feature space, but for spatial data, synthetic hexagons do not correspond to real geographic locations. Class weighting preserves the spatial integrity of the dataset.

### Figures/Tables:
| ID | Type | Description | Source Cell |
|----|------|-------------|-------------|
| Fig. 4 | Map | Spatial CV fold assignments — 5 colors on hex grid showing fold membership | Notebook ML, Section 7 |
| Eq. 1 | Formula | SpatialKFold partitioning equation | Section 7 markdown |

---

## 4. Results (~450 words)

**Narrative arc**: Which model won → What drives the predictions → How confident are we.

### Content:

#### 4.1 Model Comparison
- Present ROC-AUC (mean +/- std across 5 spatial folds) for all three models.
- Discuss the performance gap between LR (expected: ~0.65-0.75) and tree-based models (expected: ~0.75-0.85).
- Note that Spatial CV scores are lower than random CV would yield — this is by design and reflects honest generalisation performance.

#### 4.2 Confusion Matrix Analysis
- Interpret the confusion matrix of the tuned best model.
- **True Positives**: Hexes with coffee shops correctly identified.
- **True Negatives**: Hexes without coffee shops correctly rejected.
- **False Positives**: The commercially valuable recommendations (discussed in Section 5).
- **False Negatives**: Existing coffee shops the model missed — discuss why (e.g., they may be in low-population or low-connectivity areas, suggesting niche/destination shops not captured by our feature set).

#### 4.3 Feature Importance
- Present the XGBoost gain-based feature importance ranking.
- Discuss which modality dominates: expect `population` and `betweenness_centrality` to be strong (connectivity + footfall), with `level4_perc` as a demographic signal.
- Connect back to the business logic: the model has "learned" that coffee shops cluster near transit hubs (betweenness), educated populations (level4), and foot traffic (population).

#### 4.4 Logistic Regression Coefficients (Interpretability Check)
- Even if LR is not the best model, present its coefficients as an interpretability baseline.
- Positive coefficients for `population`, `level4_perc`, `n_synergy` validate the economic intuition.
- Negative coefficient for `retired_perc` or `no_qualifications_perc` would confirm demographic targeting.

### Figures/Tables:
| ID | Type | Description | Source Cell |
|----|------|-------------|-------------|
| Table 2 | Results | Model comparison: LR vs RF vs XGB — AUC, Precision, Recall, F1 (spatial CV) | Notebook ML, Section 9 |
| Fig. 5 | Chart | ROC curves — 3 models overlaid with AUC in legend | Notebook ML, Section 11a |
| Fig. 6 | Chart | Feature importance bar chart (horizontal, ranked) | Notebook ML, Section 11c |
| Fig. 7 | Matrix | Confusion matrix — tuned XGBoost | Notebook ML, Section 11b |
| Table 3 | Coefficients | Logistic Regression coefficients with 95% CI | Notebook ML, Section 8 (extended) |

---

## 5. Business Recommendations (~300 words)

**Narrative arc**: What the model recommends → Why we trust these recommendations → What they look like on the ground.

### Content:

#### 5.1 The False Positive Thesis
- Restate the key insight: FP hexagons have the learned feature profile of successful coffee shop locations but lack any current supply. These are Burt's structural holes, now validated by supervised learning rather than heuristic scoring.
- Quantify: "The model identified N False Positive hexagons out of M total hexes (X%)."

#### 5.2 Top 5 Recommended Sites
- Present a table of the top 5 FP hexes ranked by predicted probability.
- For each, provide: H3 index, latitude/longitude, population, Level 4 qualification %, betweenness centrality, number of synergy nodes.
- Add a sentence of geographic context for each (e.g., "This hex covers the intersection of [Street A] and [Street B], adjacent to [Landmark], within 200m of [Station]"). Cross-reference using Google Maps or OSM.

#### 5.3 Demographic Profile of Recommended Sites
- Aggregate demographics across the top 10 FP hexes and compare against Camden averages.
- Expected: recommended sites have above-average Level 4 %, above-average age 16-34 %, above-average population, and high betweenness centrality.

### Figures/Tables:
| ID | Type | Description | Source Cell |
|----|------|-------------|-------------|
| Fig. 8 | 3D Map | Pydeck recommendation map — green extruded hexes (FP), red flat (TP), grey (TN), orange (FN) | Notebook ML, Section 13 |
| Table 4 | Sites | Top 5 recommended sites with coordinates, features, and geographic context | Notebook ML, Section 12 + manual annotation |

---

## 6. Limitations & Future Work (~100 words)

### Content:
- **Single-borough scope**: Results are specific to Camden. Transferability to other London boroughs or cities is untested.
- **Temporal snapshot**: All data represents a single point in time (2021 Census, 2023 LandScan, 2026 OSM). Coffee shop openings/closings since data collection are not captured.
- **No revenue ground truth**: The target variable is presence/absence, not profitability. A hex with a struggling coffee shop is labelled the same as one with a thriving chain.
- **Feature limitations**: No real-time footfall data (e.g., mobile phone signals), no rental cost data, no competitor quality differentiation (specialty vs. generic).
- **Future work**: Extend to multi-class (cafe, restaurant, gym), incorporate temporal dynamics (opening/closing dates), add commercial rent data from VOA.

---

## References

**Required citations** (minimum — add more as appropriate):

1. Boeing, G. (2017). OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks. *Computers, Environment and Urban Systems*, 65, 126-139.
2. Burt, R.S. (1992). *Structural Holes: The Social Structure of Competition*. Harvard University Press.
3. Moreno, C., Allam, Z., Chabaud, D., Gall, C., & Pratlong, F. (2021). Introducing the "15-Minute City": Sustainability, resilience and place identity in future post-pandemic cities. *Smart Cities*, 4(1), 93-111.
4. Tobler, W.R. (1970). A computer movie simulating urban growth in the Detroit region. *Economic Geography*, 46(sup1), 234-240.
5. Uber Technologies (2018). H3: Uber's Hexagonal Hierarchical Spatial Index. https://h3geo.org/
6. Bright, E.A., Rose, A.N., & Urban, M.L. (2023). LandScan Global Population Database. Oak Ridge National Laboratory.
7. Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825-2830.
8. Chen, T. & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD '16*, 785-794.

---

## Appendix Checklist

Items to include in the submission appendix:

- [ ] Full feature correlation matrix (high-resolution)
- [ ] GridSearchCV results table (all parameter combinations)
- [ ] Top 20 False Positive hexes (extended table)
- [ ] Spatial CV fold map (full-page)
- [ ] AI Collaboration Audit Log (`agent_collaboration_log.md`)
- [ ] `requirements.txt` for reproducibility
- [ ] Link to GitHub repository

---

*This outline maps directly to the `camden_predictive_model.ipynb` notebook. Every figure and table references a specific notebook section for traceability.*
