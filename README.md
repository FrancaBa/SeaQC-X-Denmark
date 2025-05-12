# Quality Check (QC) for Tide Gauge Measurements in Greenland

In the scope of the GrønSL project, this repository has been developed to have an automated quality check algorithm for sea level data in Greenland including traditional machine learning approaches.

## Goal
The goal of this work is to generate an automated QC algorithm which can detect faulty measurements in timeseries with a focus on small-scale anomalies. The first version focuses on classifying sea level measurements in Greenland via a combination of statistical and traditional ML tests.

## Motivation
Quality checking of many time series is at the moment only sparsly done and often manually. However, with the increasing amount of data, it is important to have an automated approach to assess the quality of measurements/recieved values - as a model can only be as accurate as the input. Thus, it is important to have cleaned input data especially in regions with little measurements as Greenland. This work will focus on developing an automated quality check algorithm for tidal sea level data. Different methods to detect small-scale anomalies are tested and can be turned on/off by the user based on needs.

## Roadmap
This project started off by flagging tide gauge measurements into adequate groups by using basic assessment and common oceanographic packages. However, this only manages to detect faulty values to a certain degree. Therefore, a supervised ML algorithm based on manually labelled data is added to mark the left-over faulty values. The focus is set on Greenlandic data during the whole work.

## Output
Tge script returns a report describing the various QC test outcomes and a labelled timeseries in .csv with bitmask as well as QC flag.

## Overview
This QC algorithm contains a lot of different (partically overlapping) steps. In the config.json, a user can decide to turn the various steps on and off based on their needs. The various test are listed and described below. The approach of 'better marking too much than too little' is taken. Each test leads to a mask where the respectiv column is set to 1 meaning the condition is present, and the column set to 0 meaning the condition is absent. All masks for each QC step are pooled together, to a so called bitmask. A bitmask is a compact way to store and represent multiple conditions using a single integer value. Here, it is worked with 18 bits:

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

When a data point has multiple issues, the corresponding bits are set to 1, and the flags are combined using the bitwise OR operation. For example, if a data point is both a stuck value and a global outlier, you would combine the flags as follows:

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
9. missing data in timeseries, but timestamp is there (Bit 9)

There is config.json containing the used IOC flags and the bitmask used for each QC test. Users can adapt the wanted flags in the config.json and then in the corresponing source code assign certain bitmasks to a flag. For now, all the spike test outcomes are merged into one spike bitmask. The code could be adapted to have a bitmask per spike detection method

## Applied Quality Tests
The QC algorithm consists of a raw measurement mode and detided series mode. Based on the users input via the config.json, the QC steps are carried out for each time series (raw & detided) once. The removal of the tidal signal is carried out via the UTide package. All thresholds are defined and can be adapted in the config.json.

The following steps are part of the quality control:
1. Format & date control 
2. Check for missing values
3. Re-sampling of intervals (to biggest common divisor) (f.e: if measurements in 2 min and 3 min resolution, new timestep is 1 in order to consider all measurements)
4. Stability test for constant values
5. Out-of-range (global) values 
6. Segmentation to active measurement periods - Drop short and bad periods
7. Check for linear interpolation (=constant slope)
8. Analysing change rate:
* 8.1 Extreme change rate between consecutive measurements (more than physical possible)
* 8.2 Noisy periods with a lot of stronger change points in short periods
9. Spike detection:
* 9.2 Cotede
* 9.3 Improved Cotede
* 9.4 Selene (VERY SLOW!)
* 9.5 Adapted Selene - Spline analysis
* 9.6 Harmonic spike detection (testing code, but not relevant)
* 9.7 Semi-Supervised ML on harmonic data (testing code, but not relevant)
10. Shift detection:
* 10.1 Ruptures (VERY SLOW!)
* 10.2 Statistical change detection via gradient (strong change, but no change back)
11. ML-Algorithm to detect small-scale uncertainties (supervised ML - Random Forest) (ONLY WORKS ON TIDAL SIGNAL AT THE MOMENT)
12. Probably good data (=short periods of good data between long periods of bad data)

In order to perform some of the quality check steps, filled timeseries without NaNs are needed. Thus, the following three filling appraoches are used:
1. Polynomial filled series
2. Polynomial fitting series based on polynomial filled series over 7 hours
3. Spline fitting over roughly 7 hours based on existing measurements

Different versions of the ML algorithm are tested to find the best set-up for the Greenland case. However, not applied approaches are still part of the code and can be turned on and off as liked via the config.json.
Potential ML methods are:
1. Unsupervised ML - Isolation Forest
2. Supervised ML - Random Forest
3. Supervised ML - XGBoost

## Structure of QC-Framework

The QC-Tool follows the subsequent structure:
1. Dataset is loaded as df
2. Config.json is loaded
3. Df is checked for invalid format & missing data
4. Df is converted to a homogenous timesteps (find the most common timesteps and then align the greatest common divisor)
5. Run QC tests (as described above (except step 12)) based on config.json
6. If detide mode on:
* Detide data
* run QC tests again on detided series
7. Calculate probably good periods  (step 12) based on previous QC outcomes
8. Convert QC test outcomes to Bitmasks and QC Flags
9. Extract results to csv and QC report

The code contains several print, plot and extraction methods for assessment of the various QC steps.

# Execute the Python Scripts 
This repository is structured in unittest (saved under tests) and main scripts in the source folder. All outcomes are saved in the output folder, but never committed to Gitlab. In order to run the code, the unittests needs to be executed.
Within the unittests all input data is defined and links to the respective datasets defined.

## Running unittests in command prompt
1. In command line, change working directory to point to tests.
2. Run tests in python. Several options:
* 2.1. Run a specific test in a phython file (use when actively working on code)
* 2.2. Run all tests in a specific python file (here: to run code for all stations)
* 2.3. Run all tests in test folder (recommended for more complex changes when changed methods are used by several tests)

```
cd /dmidata/users/<DMI initials>/greenland_qc
python -m unittest tests.test_greenland_measurements_qc.Test_QA_Station.test_quality_check_qaqortoq
python -m unittest tests/test_greenland_measurements_qc.py
python -m unittest
```

# Pull and Push Code
The subsequent lines are describing how to pull and push changes to this repository.

## Pull code from Gitlab
1. Add a new branch on Gitlab via +-Icon
* Naming convension: initials + month abbr. + number edit in this month 
* f.e: fb_sep1, fb_sep2, ...
2. Pull code to local server (you will need to log in to your Gitlab account)
```
cd /dmidata/users/<DMI initials>
git clone --branch <branch_name> https://gitlab.dmi.dk/frb/qc_sl_greenland greenland_qc
cd greenland_qc
conda activate qc_env
code .
```
3. Activate the correct environment (here: conda environment called qc_env)

## Push altered code back to gitlab

Following steps will allow you to check your changes/improvements and push them to Gitlab.
```
git add * --dry-run
git add *
git commit -m "description of what you changed/added to code"
git push origin <branch_name>
```
After that you also need to add your appreviation and password before the push is successfull.

## Merge branch on Gitlab
Open the gitlab repository online and manually merge the branch into main. Be aware of potential conflicts. After merging a branch, make sure to delete it from the server/local machine and Gitlab.

## Contributing
Contributions to this project are welcome. Please fork the repository, make your changes, and submit a merge request with a clear description of your modifications.

# Authors and acknowledgment
This project is carried out by Franca Bauer under the supervision of Jian Su. By question, please feel free to reach out to frb@dmi.dk or jis@dmi.dk.

## Project status
The preprocessing of the data through statistical tools and existing python scripts is done. Now, it is looked into how to flag undetected measurement errors using supervied machine learning.
