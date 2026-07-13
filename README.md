# COMP9444 Galaxy Morphology Classification
Private group repository for the UNSW COMP9444 Project 057: Galaxy Morphology Classification.
## Project Status
Repository initialization only. Dataset processing and model implementation have not started.
## Repository Structure
- `data/`: local raw data and version-controlled dataset split information
- `notebooks/`: exploration, visualization, and experiment reporting
- `src/preprocessing/`: reusable data loading and preprocessing code
- `src/baseline/`: baseline implementation
- `src/models/`: model implementations, to be selected by the model owners
- `src/evaluation/`: shared evaluation and model-comparison code
- `reports/`: literature review, report, and presentation files
- `outputs/`: generated checkpoints, figures, logs, and result files
## Data Workflow
- Raw Galaxy10 DECaLS data will be stored locally under `data/raw/`.
- Raw dataset files must not be committed to Git.
- Reproducible train, validation, and test split files will be stored under `data/splits/`.
- Reusable project logic belongs in `src/`.
- Notebooks are for exploration, visualization, and reporting only. Core project logic should not be implemented in notebooks.
## Dataset Setup
Dataset acquisition instructions will be added after the download workflow has been implemented and validated.
## Team Members
Team member details will be added by the group.
