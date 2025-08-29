# Quality Check (QC) for Tide Gauge Measurements in Denmark

In the scope of this project, this repository has been developed to have an automated quality check algorithm for sea level data in Denmark including traditional machine learning approaches as one step of the QC method.

## Goal
The goal of this work is to generate an automated QC algorithm which can detect faulty measurements in time series with a focus on small-scale anomalies. The first version focused on classifying sea level measurements in Greenland via a combination of statistical and traditional ML tests. The first traditional ML model was trained and evaluated for small-scale anomalies in Greenland. Afterwards, the script was also evaluated on Danish tide gauge station data. There the training data set has been expended to all type of quality errors and it was evaluated how good the ML-tool performs overall. The conclusion is that it would make sense to combine the ML tool with statistical test for now.

## Motivation
Quality checking of many time series is at the moment only sparsely done and often manually. However, with the increasing amount of data, it is important to have an automated approach to assess the quality of measurements/received values - as a model can only be as accurate as the input. Thus, it is important to have cleaned input data. This work will focus on developing an automated quality check algorithm for sea level data. Different methods to detect data issues are tested and can be turned on/off by the user based on needs.

## Road map
This project started off by flagging tide gauge measurements into adequate groups by using basic assessment and common oceanographic packages. However, this approach only manages to detect faulty values to a certain degree. Therefore, a supervised ML algorithm based on manually labeled data is added to mark the left-over faulty values. The ML-tool on its own is good in detecting various erroneous data points, but there is still opportunities in advancing its performance. First steps would be to re-evaluate the feature selection and expand the training data set.

## Output
The script returns a report describing the various QC test outcomes and a labeled time series in .csv with bit mask as well as QC flag.

## Overview
This QC algorithm contains a lot of different (partially overlapping) steps. In the config.json file, a user can decide to turn the various steps on and off based on their needs. The various test are listed and described below. The approach of 'better marking too much than too little' is taken. Each test leads to a mask where the respective column is set to 1 meaning the condition is present, and the column set to 0 meaning the condition is absent. All masks for each QC step are pooled together, to a so called bit mask. A bit mask is a compact way to store and represent multiple conditions using a single integer value. Here, it is outlined with 18 bits:

- Bit 0: Shifted period
- Bit 1: Noisy data period
- Bit 2: Probably good data
- Bit 3: Stuck value
- Bit 4: Global outlier
- Bit 5: Spike
- Bit 6: Incorrect format
- Bit 7: Bad segment
- Bit 8: Interpolated value
- Bit 9: Missing data
- Bit 10: ML detected anomalies
- Bit 11: Shifted period detided
- Bit 12: Noisy data period detided
- Bit 13: Stuck value detided
- Bit 14: Global outlier detided
- Bit 15: Spike detided
- Bit 16: Interpolated data detided
- Bit 17: ML detected anomalies detided

Each of these conditions can be represented as a specific bit in a 18-bit binary number. The bits are assigned the following powers of 2:

- Bit 0 (Shifted period): 00 0000 0000 0000 0001 → 1
- Bit 1 (Noisy data period): 00 0000 0000 0000 0010 → 2
- Bit 2 (Probably good data): 00 0000 0000 0000 0100 → 4
- Bit 3 (Stuck value): 00 0000 0000 0000 1000 → 8
- Bit 4 (Global outlier): 00 0000 0000 0001 0000 → 16
- Bit 5 (Spike): 00 0000 0000 0010 0000 → 32
- Bit 6 (Incorrect format): 00 0000 0000 0100 0000 → 64
- Bit 7 (Bad segment): 00 0000 0000 1000 0000 → 128
- Bit 8 (Interpolated data): 00 0000 0001 0000 0000 → 256
- Bit 9 (Missing data): 00 0000 0010 0000 0000 → 512
- Bit 10 (ML detected anomalies): 00 0000 0100 0000 0000 → 1024
- Bit 11 (Shifted period detided): 00 0000 1000 0000 0000 → 2048
- Bit 12 (Noisy data period detided): 00 0001 0000 0000 0000 → 4096
- Bit 13 (Stuck value detided): 00 0010 0000 0000 0000 → 8192
- Bit 14 (Global outlier detide): 00 0100 0000 0000 0000 → 16384
- Bit 15 (Spike detided): 00 1000 0000 0000 0000 → 32768
- Bit 16 (Interpolated data detided): 01 0000 0000 0000 0000 → 65536
- Bit 17 (ML detected anomalies detided): 10 0000 0000 0000 0000 → 131072

When a data point has multiple issues, the corresponding bits are set to 1, and the flags are combined using the bit wise OR operation. For example, if a data point is both a stuck value and a global outlier, you would combine the flags as follows:

- Stuck Value(3) | Global Outlier (4): 00 0000 0000 0001 1000 → 24

Based on the bitmask, IOC flags are assigned to the measurement point. F.e. missing data equals flag 9 in IOC.

0. no quality check carried out
1. good_data (no bit)
2. probably good data (Bit 2)
3. bad data, but correctable through other parameters or knowledge (Bit 1, Bit 12)
4. bad data (Bit 3, Bit 4, Bit 6, Bit 7, Bit 13 and Bit 14)
5. shifted value (= period of series is offset towards to mean) (Bit 0, Bit 11)
6. spikes (= smaller local outlier) (Bit 5, Bit 15)
7. ML-detected anomalies (= small scale anomalies detected by ML algorithm) (Bit 10 and Bit 17)
8. linear interpolated value (Bit 8, Bit 16)
9. missing data in time series, but timestamp is there (Bit 9)

There is config.json containing the used IOC flags and the bit mask used for each QC test. Users can adapt the wanted flags in the config.json and then in the corresponding source code assign certain bit masks to a flag. For now, all the spike test outcomes are merged into one spike bit mask. The code could be adapted to have a bit mask per spike detection method.

## Applied Quality Tests
The QC algorithm consists of a raw measurement mode and detided series mode. Based on the users input via the config.json, the QC steps are carried out for each time series (raw & detided) once. The removal of the tidal signal is carried out via tidal time series as additional input parameter. In the first version, detiding with the python package 'UTide' was tested, but it didn't work as accurate as using tidal signals directly and has been therefore discarded. All thresholds for the statistical tests are defined and can be adapted in the config.json. The advantage of the ML-method is that it doesn't need any thresholds. Only one threshold is needed to convert the probability outcome of the ML-model to a true/false. This parameter is also set in the config.json.

The following steps are part of the quality control:
1. Format & date control 
2. Check for missing values
3. Re-sampling of intervals (to biggest common divisor) (f.e: if measurements in 2 min and 3 min resolution, new time step is 1 in order to consider all measurements)
4. Stability test for constant values
5. Out-of-range (global) values 
6. Segmentation to active measurement periods - Drop short and bad periods (f.e: only 1 measurement within 24 hr)
7. Check for linear interpolation (=constant slope)
8. Analyzing change rate:
* 8.1 Extreme change rate between consecutive measurements (more than physical possible)
* 8.2 Noisy periods with a lot of stronger change points in short periods
9. Spike detection:
* 9.2 Cotede
* 9.3 Improved Cotede
* 9.4 Like Selene (VERY SLOW!)
* 9.5 Adapted Selene - Spline analysis (much faster and as good as Selene)
deviates tot much - basically ML-version of Selene))
10. Shift detection: (None of the methods work sufficiently! It is not recommended to use them.)
* 10.1 Ruptures (VERY SLOW!)
* 10.2 Statistical change detection via gradient (strong change, but no change back)
11. ML-Algorithm to detect small-scale uncertainties (supervised ML - Random Forest) (ONLY WORKS ON TIDAL SIGNAL AT THE MOMENT)
12. Probably good data (=short periods of good data between long periods of bad data)

In order to perform some of the quality check steps, filled time series without NaNs are needed. Thus, the following three filling approaches are used:
1. Polynomial filled series
2. Polynomial fitting series based on polynomial filled series over 7 hours
3. Spline fitting over roughly 7 hours based on existing measurements

Different versions of the ML algorithm have tested to find the best set-up for the Greenland case and the best model afterwards applied on the Danish stations. However, not used methods are still part of the code and can be commented in or out as wanted. The ML spike detection runs now with a RandomForest model.

Tested ML methods with different training approaches and features are:
1. Unsupervised ML - Isolation Forest
2. Supervised ML - Random Forest
3. Supervised ML - XGBoost


## Structure of QC-Framework

The QC-Tool follows the subsequent structure:
1. Dataset is loaded as df
2. Config.json is loaded
3. DF is checked for invalid format & missing data
4. DF is converted to a homogenous timesteps (find the most common timesteps and then align the greatest common divisor)
5. Run QC tests (as described above (except step 12)) based on config.json
6. If detide mode on:
* Detide data
* Run QC tests again on detided series
* No ML test on detided data for now
7. Calculate probably good periods  (step 12) based on previous QC outcomes
8. Convert QC test outcomes to Bitmasks and QC Flags
9. Extract results to csv and QC report

The code contains several print, plot and extraction methods for assessment of the various QC steps.

# Execute the Python Scripts 
This repository is structured in unittest (saved under tests) and main scripts in the source folder. All outcomes are saved in the output folder, but never committed to GitHub. In order to run the code, the unittests needs to be executed.
Within the unittests all input data is defined and links to the respective datasets defined. The source folder contains the main qc method 'main.py' which runs the QC control by calling all other scripts and the respective QC tests in 'various_qc_tests'. 

## Running unittests in command prompt
1. In command line, change working directory to point to tests.
2. Run tests in python. Several options:
* 2.1. Run a specific test in a phython file (use when actively working on code) (this might run seconds to minutes)
* 2.2. Run all tests in a specific python file (here: to run code for all stations) (this might run seconds to minutes)
* 2.3. Run all tests in test folder (recommended for more complex changes when changed methods are used by several tests or before a commit to GitHub) (this can take up to several hours depending on which QC tests are active.)

```
cd /home/documents/greenland_qc
python -m unittest tests.test_ml_outlier.Test_QC_ML_Station.test_quality_check_ml_DK
python -m unittest tests/test_ml_outlier.py
python -m unittest
```

# Set the correct environment
The GitHub directory contains a file 'environment.yml' which contains the revent environment set-up needed to run the script. When using this script for the first time, the environment can be generated in the following way:
```
conda env create -f environment.yml
conda activate qc_env
```

# Pull and Push Code
The subsequent lines are describing how to pull and push changes to this repository.

## Pull code from GitHub
1. Add a new branch on GitHub via +-Icon
* Naming convension: initials + month abbr. + number edit in this month 
* f.e: fb_aug1, fb_aug2, ...
2. Pull code to local server (you will need to log in to your GitHub account)
* Pull from GitHub
```
cd /home/documents/greenland_qc
git clone --branch <branch_name> https://github.com/FrancaBa/SeaQC-X-Denmark ml_qc_denmark
cd ml_qc_denmark
```

3. Activate the correct environment (here: conda environment called qc_env)
```
conda activate qc_env
```
4. Open Python in wanted GUI. Here f.e. in Visual Studio code:
```
code .
```
## Push altered code back to GitHub

Following steps will allow you to check your changes/improvements and push them to GitHub.
```
git add * --dry-run
git add *
git commit -m "description of what you changed/added to code"
git push origin <branch_name>
```
After that you also need to add your 
## Structure of QC-Framework

The QC-Tool follows the subsequent structure:
1. Dataset is loaded as df
2. Config.json is loaded
3. DF is checked for invalid format & missing data
4. DF is converted to a homogenous timesteps (find the most common timesteps and then align the greatest common divisor)
5. Run QC tests (as described above (except step 12)) based on config.json
6. If detide mode on:
* Detide data
* Run QC tests again on detided series
* No ML test on detided data for now
7. Calculate probably good periods  (step 12) based on previous QC outcomes
8. Convert QC test outcomes to Bitmasks and QC Flags
9. Extract results to csv and QC report

The code contains several print, plot and extraction methods for assessment of the various QC steps.

# Execute the Python Scripts 
This repository is structured in unittest (saved under tests) and main scripts in the source folder. All outcomes are saved in the output folder, but never committed to GitHub. In order to run the code, the unittests needs to be executed.
Within the unittests all input data is defined and links to the respective datasets defined. The source folder contains the main qc method 'main.py' which runs the QC control by calling all other scripts and the respective QC tests in 'various_qc_tests'. 

## Running unittests in command prompt
1. In command line, change working directory to point to tests.
2. Run tests in python. Several options:
* 2.1. Run a specific test in a phython file (use when actively working on code) (this might run seconds to minutes)
* 2.2. Run all tests in a specific python file (here: to run code for all stations) (this might run seconds to minutes)
* 2.3. Run all tests in test folder (recommended for more complex changes when changed methods are used by several tests or before a commit to GitHub) (this can take up to several hours depending on which QC tests are active.)

```
cd /home/documents/greenland_qc
python -m unittest tests.test_ml_outlier.Test_QC_ML_Station.test_quality_check_ml_DK
python -m unittest tests/test_ml_outlier.py
python -m unittest
```

# Set the correct environment
The GitHub directory contains a file 'environment.yml' which contains the recent environment set-up needed to run the script. When using this script for the first time, the environment can be generated in the following way:
```
conda env create -f environment.yml
conda activate qc_env
```

# Pull and Push Code
The subsequent lines are describing how to pull and push changes to this repository.

## Pull code from GitHub
1. Add a new branch on GitHub via +-Icon
* Naming convention: initials + month abbr. + number edit in this month 
* f.e: fb_aug1, fb_aug2, ...
2. Pull code to local server (you will need to log in to your GitHub account)
* Pull from GitHub
```
cd /home/documents/greenland_qc
git clone --branch <branch_name> https://github.com/FrancaBa/SeaQC-X-Denmark ml_qc_denmark
cd ml_qc_denmark
```

3. Activate the correct environment (here: conda environment called qc_env)
```
conda activate qc_env
```
4. Open Python in wanted GUI. Here f.e. in Visual Studio code:
```
code .
```
## Push altered code back to GitHub

Following steps will allow you to check your changes/improvements and push them to GitHub.
```
git add * --dry-run
git add *
git commit -m "description of what you changed/added to code"
git push origin <branch_name>
```
After that you also need to add your abbreviation and password before the push is successful.

## Merge branch on GitHub
Open the GitHub repository online and manually merge the branch into main. Be aware of potential conflicts. After merging a branch, make sure to delete it from the server/local machine and GitHub.

## Contributing
Contributions to this project are welcome. Please fork the repository, make your changes, and submit a merge request with a clear description of your modifications. and password before the push is successful.

## Merge branch on GitHub
Open the GitHub repository online and manually merge the branch into main. Be aware of potential conflicts. After merging a branch, make sure to delete it from the server/local machine and GitHub.

## Contributing
Contributions to this project are welcome. Please fork the repository, make your changes, and submit a merge request with a clear description of your modifications.

