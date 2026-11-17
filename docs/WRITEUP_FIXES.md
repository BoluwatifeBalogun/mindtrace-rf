# Corrections needed in the write-up (Chapters 1 to 3)

The "Corrected" version still carries text from the earlier CNN / VGG / Grad-CAM drafts.
Most of it comes from replacing the words "CNN" or "deep learning" with "Random Forest"
without rewriting the sentence, which leaves statements that are technically false.
A panel member who knows the algorithm will notice these first.

## A. Statements that are now technically wrong (fix before submission)

| # | Where | Problem | Fix |
| --- | --- | --- | --- |
| 1 | 1.1 (paras 5 to 7), 2.9 | Random Forest is described as learning "hierarchical visual features" through "successive layers", with "pooling operations", "ReLU" and "convolutional" steps. Those describe a CNN. | Describe it as in 2.7: an ensemble of decision trees trained on a numeric feature vector, with bootstrap sampling and random feature selection (Breiman, 2001). Delete 2.9 or merge its last two paragraphs into 2.7. |
| 2 | 2.8 "Random Forest training", 1.7 definition, 1.6, 2.4, 2.5, 3.15.3 | This is the old **transfer learning** section. It says a model that "previously learned general visual features" is adapted. 2.7 correctly says no pretrained model or transfer learning is used, so the document contradicts itself. | Rewrite 2.8 as "Training a Random Forest": bootstrap samples, random subset of features at each split, majority vote, out-of-bag idea, hyperparameters. Remove the 1.7 definition. |
| 3 | 1.1, 2.3, 2.4, 2.5, 2.6, 2.7, 2.15 | Kaur et al. (2024), Zia-Ur-Rehman et al. (2024) and the "brain MRI review" are quoted as finding that "Random Forest models, including 3D Random Forest" dominate and beat "conventional machine-learning methods". Those reviews said that about CNNs and deep learning. Random Forest is itself conventional machine learning, so the sentences misreport the sources. | Say what the reviews found (deep learning, especially CNNs, dominates recent work) and then justify Random Forest as the simpler, more transparent alternative suited to this project, as 2.14.1 already does. |
| 4 | 1.1 | "feature-importance information, introduced by Selvaraju et al. (2017) ... gradients flowing into a convolutional layer". That is Grad-CAM, and Selvaraju is not in the reference list. | Replace with Random Forest variable importance (Breiman, 2001): the average decrease in impurity a feature produces across all trees. |
| 5 | 1.2, 2.12.2, 2.14.3, 3.12.9, 3.12.12, 3.12.13, 3.18.4 | Feature importance is called "visual", said to "highlight regions that influenced the prediction". 2.2.5 and 2.10 correctly say it is not a heatmap and does not locate brain regions. | Use one description everywhere: "a ranked list of the texture and intensity features the model relies on most". |
| 6 | Table 3.3, 3.15.3 | "NVIDIA GPU with CUDA ... accelerating Random Forest training". scikit-learn's Random Forest runs on the CPU. | Remove the GPU row. Training took about 4 minutes on CPU. |
| 7 | 2.2.4 | "The final classification layer will generate a probability". A forest has no layers. | "The forest averages the votes of its trees to give a class probability." |
| 8 | 3.16.2 | Data augmentation "to reduce overfitting" is a deep-learning practice and is not used. | Delete the sentence. |

## B. Leftover wording

- "Machine Learning and Machine Learning" (title of 2.6, also 1.1, 1.8, 2.1, 3.11). Use "Machine Learning".
- Heading 2.2.3 "Deep-Learning Model" -> "Random Forest Model". Software table: "scikit-learn | Deep-learning model development" -> "Machine-learning model development and evaluation".
- Abbreviations: "DL | Machine Learning" (remove), "Random Forest | Random Forest classifier" (use RF), one row with an empty abbreviation, "feature importance | Feature Importance" (remove), GPU (remove). Add GLCM and RF.
- 1.7 defines "Random Forest classifier" twice and "Machine Learning" twice; the first "Machine Learning" definition is the old deep-learning one.
- "predicted class and, where supported, class probability" appears about 30 times. Random Forest always gives a probability, so write "predicted class and class probability".
- "Confidence Score" (3.12.8, 3.12.9, 3.18) vs "class probability". Pick "class probability".
- 2.9: "without relying on a pretrained previous deep-learning architectures model or feature-based learning". The system *is* feature-based. Delete the clause.
- 3.12.3: "The model will uses". 3.6: "unusable images were identified" -> "images identified". 3.16.5: stray sentence "The system will contribute to the prediction." 2.2.7: stray line "Web-Based Results". 2.10: the same paragraph appears twice.
- TC08 expected result is just "is produced". Write "Ranked feature-importance list is produced".
- Table 3.4 has an empty row.

## C. Numbering and front matter

- Missing section numbers: 2.11, 3.12.5, 3.13.5, 3.16.6, 3.17.7. 3.13.10 and 3.15.8 are typed as body text, and 3.15.7 has no content.
- Table of contents omits 3.8.3, 3.12.10 and 3.12.11, lists several entries without page numbers, and still titles 2.6 "Machine Learning and Machine Learning".
- Table 2.1 appears after the chapter summary (2.16). Move it into 2.13.

## D. Align the text with what was built

| Topic | Write-up says | Implementation | Suggested wording |
| --- | --- | --- | --- |
| Dataset | ADNI is "the primary dataset under consideration" | ADNI needs an approved application. Used the public Alzheimer MRI dataset: 6,400 axial slices, 128 x 128, four labels | State the dataset and the mapping: No AD = Non-demented (3,200); AD = Very mild + Mild + Moderate (3,200). Keep ADNI as background only. |
| Subject leakage (3.5, 3.10.2) | Slices of one subject will be kept in one subset | The dataset has no subject identifiers, so this could not be enforced | Say so, and list it as a limitation in Chapter 5. |
| Split | Not specified | 70 / 15 / 15 stratified by stage, seed 42: 4,480 / 960 / 960 | Add to 3.5. |
| Features | "exact features will be finalized" | 69: 42 GLCM (7 descriptors, distances 1, 2, 4, mean and directional spread over 4 angles, 32 gray levels) and 27 intensity statistics | Add to 3.12.3 or Chapter 4. |
| Image size | "required input dimensions" | 128 x 128, min-max normalised to 0 to 255 | Add to 3.12.2. |
| Metrics | ROC-AUC "where appropriate" | Reported, with specificity and 5-fold cross-validation | Add both to 3.17. |
| Database | MySQL | MySQL through `DATABASE_URL`; SQLite fallback for the free demo host | Mention the fallback in Chapter 4. |
| Users | "authorized user" | Self-registration with role, facility and state; scan reference code instead of patient name | Add to 3.12.1. |
| Case study | 3.3 says no physical study area; 1.1 quotes only United States figures | App is themed and structured for Nigeria (36 states + FCT, six zones) | Add a short paragraph on dementia care in Nigeria with a real citation from your library. I did not add any figure for you because I could not verify one. |

## E. Figures for Chapter 4 (from `models/metrics.json`)

Accuracy 81.7%, precision 78.4%, recall 87.5%, F1 82.7%, specificity 75.8%, ROC-AUC 0.906.
Confusion matrix: TP 420, FN 60, FP 116, TN 364 (960 test images).
5-fold cross-validation accuracy 82.8% ± 1.5. Training accuracy 100%, validation 83.3%.
By stage: Non-demented 364/480, Very mild 278/336, Mild 133/135, Moderate 9/9.
Average processing time per image: about 25 to 40 ms on a laptop CPU.
