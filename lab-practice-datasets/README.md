# Practice datasets for the plotting lab

Eight synthetic datasets, one per figure type in this toolkit. The values are
simulated for teaching and are not real measurements.

| # | File | Figure / script | What the data is | What to enter |
| --- | --- | --- | --- | --- |
| 1 | `1_bar-chart_concrete_compressive_strength.xlsx` | Bar chart: `bar-chart-demo/bar_chart.py` | How much pressure (MPa) 6 different concrete recipes can take before breaking. Each recipe was tested 4 times. | x label: `Mix design`, y label: `Compressive strength (MPa)` |
| 2 | `2_scatter-chart_coffee_NIR_PCA.xlsx` | Scatter: `scatter-chart-demo/scatter-chart.py` | Each point is one coffee bean sample from one of 5 countries. `PC1` and `PC2` are two scores that summarise how the sample absorbs light. | Label column: `Origin`, x: `PC1`, y: `PC2` |
| 3 | `3_histogram_particle_size_milling.xlsx` | Histogram: `histogram/histogram.py` | Size (µm) of powder particles made by 3 grinding machines, one column per machine. The columns have different lengths because a different number of particles was measured for each machine. | Columns: press Enter to select all. Bins: start with 40 |
| 4 | `4_predicted-vs-actual_battery_SOH.xlsx` | Predicted vs actual: `predicted-actual-plot/predicted_vs_actual.py` | Battery health: how much charge a used battery can still hold, as a % of a new one. Two models predicted this value. | Actual: `Actual_SOH_percent`. Predicted: run once with `Predicted_SOH_RandomForest` and once with `Predicted_SOH_LinearRegression` |
| 5 | `5_AUC-ROC_sepsis_prediction.csv` | ROC/AUC: `AUC-ROC/auc_roc.py` | Hospital patients. `true_label` is 1 if the patient developed sepsis (a severe reaction to infection) and 0 if not. The other 5 columns are each model's predicted probability (0–1). | Only the file name |
| 6 | `6_boxplot_3Dprint_tensile_strength.csv` | Boxplot: `boxplot/visualize_boxplot.py` | Strength (MPa) of 3D-printed plastic parts: how hard you must pull before the part breaks. The parts were printed with 5 different internal structures, listed in `infill_pattern`. | Category: `infill_pattern`, value: `tensile_strength_MPa`. Axis titles: `Infill pattern`, `Tensile strength (MPa)` |
| 7 | `7_confusion-matrix_land_cover.csv` | Confusion matrix: `confusion-matrix/confusion_matrix.py` | Satellite image patches sorted into 6 land types. `true_label` is the real land type and `predicted_label` is the model's answer. | Press Enter to accept the default column names. Optional: add `--labels Water Forest Grassland Cropland Urban Bare_soil` to fix the class order |
| 8 | `8_heat-map_water_quality_correlation.csv` | Heatmap: `heat-map/visualize_heatmap.py` | How strongly pairs of 9 river water measurements change together, from −1 to 1. Positive means both rise together; negative means one rises while the other falls. | Row title: `Parameter`, column title: `Parameter` |

File formats: the bar-chart and scatter scripts read only Excel files, and the
ROC, boxplot, confusion-matrix and heatmap scripts read only CSV files. The
histogram and predicted-vs-actual scripts read either format.

Example figures: each dataset has one or more `.png` files with the same name,
showing what the finished figure should look like.

## Questions to try

1. **Bar chart:** Which mixes do the significance letters show as no different from the control?
2. **Scatter:** Which two origins overlap most? What might that mean for a classifier?
3. **Histogram:** Which milling method gives the smallest particles, and which gives the widest spread of sizes?
   Does any method show more than one peak? Re-plot with 15, 40 and 80 bins: which bin count makes the shape easiest to see?
4. **Predicted vs actual:** Compare R²/RMSE for the two models. Which model is better?
5. **ROC:** Rank the models by AUC.
6. **Boxplot:** Which group has the highest median, and which has the lowest?
7. **Confusion matrix:** Which class is predicted best, and which is predicted worst?
8. **Heatmap:** Which parameters are positively and negatively correlated with dissolved O₂?
