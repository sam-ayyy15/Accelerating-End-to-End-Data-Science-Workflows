# Accelerating End-to-End Data Science Workflows - NVIDIA Certification

![NVIDIA Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/NVIDIA_logo.svg/320px-NVIDIA_logo.svg.png)

## 📝 Course Description
This intensive NVIDIA certification program focuses on **GPU-accelerated data science** using the RAPIDS ecosystem. The course covers end-to-end workflows from data processing to machine learning deployment, demonstrating how GPUs can dramatically accelerate data science pipelines.

## 🎯 Learning Objectives
- Accelerate data processing with **cuDF** (10-100x faster than pandas/Polars)
- Implement GPU-accelerated machine learning with **cuML** and **XGBoost**
- Perform large-scale graph analytics with **cuGraph**
- Deploy optimized models using **Triton Inference Server**
- Solve real-world problems with GPU-accelerated workflows

## 🛠️ Technical Stack
### Core Technologies
| Category        | Tools/Libraries |
|----------------|---------------|
| **Data Processing** | cuDF, pandas, Polars, Dask |
| **Machine Learning** | cuML, XGBoost, Scikit-learn |
| **Graph Analytics** | cuGraph, NetworkX |
| **GPU Computing** | CuPy, RAPIDS |
| **Visualization** | Bokeh, Matplotlib |
| **Deployment** | Triton Inference Server |

 # Biodefense Outbreak Analysis - UK Epidemic Simulation

![NVIDIA RAPIDS Logo](https://developer.nvidia.com/sites/default/files/akamai/rapids/images/RAPIDS-logo-white-500px.png)  
*Accelerated with NVIDIA RAPIDS*

## Project Overview
This project analyzes a simulated epidemic outbreak in the UK using GPU-accelerated data science tools. The workflow spans three critical weeks of outbreak response:

1. **Geospatial Cluster Detection** (Week 1)
2. **Healthcare Facility Optimization** (Week 2 - Coming Soon)
3. **Risk Factor Identification** (Week 3)

## Key Features
- Processes **58+ million records** with GPU acceleration
- Identifies infection hotspots using DBSCAN clustering
- Transforms geographic coordinates with custom CuPy functions
- Pinpoints high-risk demographic groups
- Optimizes healthcare resource allocation

## Week 1: Geospatial Cluster Analysis
### Methods
- Filtered 58,479,894 patient records using **cuDF**
- Implemented **DBSCAN clustering** (via cuML) with:
  - **EPS**: 2000m 
  - **Minimum samples**: 25 infected individuals
- Developed custom **lat/long → OSGB36 grid converter** in CuPy

### Results
- Identified **14 distinct clusters** across the UK
- Largest cluster:
  - **Location**: Northing 397,661 / Easting 371,410
  - **Infected**: 8,638 individuals
  - **Geographic center**: [54.52°N, -1.57°W] (North Yorkshire region)

![Cluster Visualization](images/cluster_map.png)  
*Example cluster visualization (simulated data)*

## Week 3: Risk Factor Identification
### Key Findings
| Risk Factor | Infection Rate |
|-------------|----------------|
| Healthcare Workers | 12.8% |
| Food Service Workers | 10.4% |
| Women (across all sectors) | 2-3× higher than men |

## Technologies Used
- **RAPIDS Ecosystem**:
  - cuDF (GPU-accelerated DataFrames)
  - cuML (GPU machine learning)
- **CuPy**: Custom coordinate transformations
- **Python 3.10**
- **Jupyter Notebooks**

## Installation
```bash
conda create -n biodefense -c rapidsai -c nvidia -c conda-forge \
    rapids=23.06 python=3.10 cudatoolkit=11.8
conda activate biodefense
pip install cupy-cuda11x
