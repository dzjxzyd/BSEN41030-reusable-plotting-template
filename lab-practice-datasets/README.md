# Practice datasets for the plotting lab

Eight synthetic datasets, one per figure type in this toolkit. The values are
simulated for teaching and are not real measurements.

| # | File | Figure / script | What to enter |
|---|------|-----------------|---------------|
| 1 | `1_bar-chart_concrete_compressive_strength.xlsx` | Bar chart: `bar-chart-demo/bar_chart.py` | 6 concrete mix designs × 4 replicates (MPa). x label: `Mix design`, y label: `Compressive strength (MPa)` |
| 2 | `2_scatter-chart_coffee_NIR_PCA.xlsx` | Scatter: `scatter-chart-demo/scatter-chart.py` | PCA scores of NIR spectra from coffee beans of 5 origins. Label column `Origin`, x `PC1`, y `PC2` |
| 3 | `3_histogram_particle_size_milling.xlsx` | Histogram: `histogram/histogram.py` | Particle size (µm) from 3 milling methods. Columns have different lengths (different numbers of particles measured) |
| 4 | `4_predicted-vs-actual_battery_SOH.xlsx` | Predicted vs actual: `predicted-actual-plot/predicted_vs_actual.py` | Battery state of health (%). Actual `Actual_SOH_percent`. Predicted: try both `Predicted_SOH_RandomForest` and `Predicted_SOH_LinearRegression` |
| 5 | `5_AUC-ROC_sepsis_prediction.csv` | ROC/AUC: `AUC-ROC/auc_roc.py` | Sepsis early-warning: `true_label` (0/1) plus 5 model probability columns |
| 6 | `6_boxplot_3Dprint_tensile_strength.csv` | Boxplot: `boxplot/visualize_boxplot.py` | Tensile strength of 3D-printed PLA. Category `infill_pattern`, value `tensile_strength_MPa` (contains outliers from defective prints) |
| 7 | `7_confusion-matrix_land_cover.csv` | Confusion matrix: `confusion-matrix/confusion_matrix.py` | Satellite land-cover classification: 6 imbalanced classes, `true_label` / `predicted_label`. Optional: add `--labels Water Forest Grassland Cropland Urban Bare_soil` to fix the class order |
| 8 | `8_heat-map_water_quality_correlation.csv` | Heatmap: `heat-map/visualize_heatmap.py` | Pearson correlations between 9 river water-quality parameters (signed values, so it uses a diverging colour map) |

File formats: the bar-chart and scatter scripts read only Excel files, and the
ROC, boxplot, confusion-matrix and heatmap scripts read only CSV files. The
histogram and predicted-vs-actual scripts read either format.

## Questions to try

1. **Bar chart:** Which mixes do the significance letters show as no different from the control?
2. **Scatter:** Which two origins overlap most? What might that mean for a classifier?
3. **Histogram:** Which milling method gives the smallest particles, and which gives the widest spread of sizes?
   Does any method show more than one peak? Re-plot with 15, 40 and 80 bins: which bin count makes the shape easiest to see?
4. **Predicted vs actual:** Compare R²/RMSE for the two models. Which model is better?
5. **ROC:** Rank the models by AUC.
6. **Boxplot:** Which infill pattern is the strongest, and which is the weakest?
7. **Confusion matrix:** Which class is predicted best, and which is predicted worst?
8. **Heatmap:** Which parameters are positively and negatively correlated with dissolved O₂?
