FINAL MATRIX DELIVERABLE
========================

graphs/
- confusion_matrix_val_clean.png: Confusion matrix for the final model on val_clean.
- confusion_matrix_val_clean_normalized.png: Row-normalized confusion matrix on val_clean.
- confusion_matrix_val_full.png: Confusion matrix on the original validation set.
- per_class_f1_comparison.png: F1 comparison with AffectNet+ Paper Table 9 ResNet-50.
- overall_metrics_comparison.png: Overall accuracy and Macro F1 comparison with paper baselines.
- training_curve_accuracy_f1.png: Validation accuracy and Macro F1 over training epochs.
- per_class_accuracy_over_epochs.png: Accuracy trend for minority classes across epochs.
- tau_calibration_curve.png: Effect of Logit Adjustment (Tau) on accuracy and F1.
- class_distribution_train_vs_val.png: Class imbalance comparison between Train and Val Clean.
- easy_challenging_difficult_breakdown.png: Reliability tier composition per class.
- label_agreement_duplicates.png: Overall label agreement on exact duplicates (31.2% agreement).
- calibration_reliability_diagram.png: Confidence vs Accuracy calibration diagram.

tables/
- final_metrics_val_clean.csv/.txt: Detailed metrics on val_clean.
- final_metrics_val_full.csv/.txt: Detailed metrics on val_full.
- final_summary_table.csv/.txt: Summary of all model variants on val_clean.
- paper_comparison_table.csv/.txt: Comparison against AffectNet+ paper figures.
- data_quality_findings_table.csv/.txt: Summary of data leakage and label noise findings.

raw_numbers/
- predictions_val_clean.csv: Raw predictions and class probabilities on val_clean.
- predictions_val_full.csv: Raw predictions and class probabilities on val_full.
