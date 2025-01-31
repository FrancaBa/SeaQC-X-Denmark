# Quality Check (QC) for Tide Gauge Measurements in Greenland

In the scope of the GrønSL project, this repository has been developed to have an automated quality check algorithm for sea level data in Greenland including state of the art machine learning approaches.

## Goal
The goal of this work is to generate an automated QC algorithm which can detect and adapt faulty measurements in timeseries using ML. The first version will be focusing on checking sea level measurements in Greenland.

## Motivation
Quality checking of many time series is at the moment only sparsly done and often manually. However, with the increasing amount of data, it is important to have an automated approach to assess the quality of measurements/recieved values - as a model can only be as accurate as the input. Thus, it is important to have cleaned input data especially in regions with little measurements as Greenland. This work will focus on developing an automated quality check algorithm for sea level data.

## Roadmap
This project will start off by flagging and adapting tide gauge measurements in adequate groups by using basic assessment and common oceanographic packages. However, this will only manage to detect faulty values to a certain degree. Therefore,the goal is to use a supervised ML algorithm based on manually labelled data to mark the left-over faulty values. Afterwards, the mark series should be cleaned/ interpolated to generate a sound baseline timeseries for future research. The timeline for this project is around 11 months.

## Overview

This QC algorithm contains a lot of different (partically overlapping) steps. In the config.json, a user can decide to turn the various steps on and off based on their needs. The various test are listed and described below. The approach of 'better marking too much than too little' is taken. Each test leads to a mask where the column is set to 1 meaning the condition is present, and the column set to 0 meaning the condition is absent. All masks for each QC step are pooled together, to a so called bitmask. A bitmask is a compact way to store and represent multiple conditions using a single integer value. 

This is an example and doesn't corresponds to the bits used here!!

See an example:

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

Each of these conditions can be represented as a specific bit in a 10-bit binary number. The bits are assigned the following powers of 2:

- Bit 0 (Shifted period): 00000 00001 → 1
- Bit 1 (Noisy data period): 00000 00010 → 2
- Bit 2 (Probably good data): 00000 00100 → 4
- Bit 3 (Stuck value): 00000 01000 → 8
- Bit 4 (Global outlier): 00000 10000 → 16
- Bit 5 (Spike): 00001 00000 → 32
- Bit 6 (Incorrect format): 00010 00000 → 64
- Bit 7 (Bad segment): 00100 00000 → 128
- Bit 8 (Interpolated data): 01000 0000 → 256
- Bit 9 (Missing data): 10000 00000 → 512

When a data point has multiple issues, the corresponding bits are set to 1, and the flags are combined using the bitwise OR operation. For example, if a data point is both a stuck value and a global outlier, you would combine the flags as follows:

- Stuck Value(3) | Global Outlier (4): 00000 11000 → 24

Based on the bitmask, IOC flags are assigned to the measurement point. F.e. missing data equals flag 9 in IOC.

0. no quality check carried out
1. good_data (no bit)
2. probably good data (Bit 2)
3. bad data, but correctable through other parameters or knowledge (Bit 1 and Bit 7)
4. bad data (Bit 4 and Bit 6)
5. shifted value (= period of series is offset towards to mean) (Bit 0)
6. spikes (= smaller local outlier) (Bit 5)
7. stuck value (= constant value over a time) (Bit 3)
8. linear interpolated value (Bit 8)
9. missing data in timeseries, but timestamp is there (Bit 9)

This approach is also followed in this project. There is config.json containing the used IOC flags and the bitmask used for each QC test. Users can adapt the wanted flags in the config.json and then in the corresponing source code assign certain bitmasks to a flag.

## Applied Quality Tests
The QC algorithm consists of a raw measurement and detided series mode. Based on the users input, the QC steps are carried out for each time series (raw & detided) once. The removal of the tidal signal is carried out via the UTide package. The following steps are part of the quality control:
1. Format & date control 
2. Check for missing values
3. Re-sampling of intervals
4. Stability test for constant values
5. Out-of-range values
6. Segmentation to active measurement periods - Drop short and bad periods
7. Check for linear interpolation (=constant slope)
8. Analysing change rate:
* 8.1 Extreme change rate between consecutive measurements (more than physical possible)
* 8.2 Noisy periods with a lot of stronger change points in short periods
9. Spike detection:
* 9.1 Statisitcal test
* 9.2 Cotede
* 9.3 Improved Cotede
* 9.4 Selene
* 9.5 Adapted Selene - Spline analysis
* 9.6 Harmonic spike detection
* 9.7 Supervised ML on harmonic (not really relevant)
10. Shift detection:
* 10.1 Ruptures
* 10.2 Statistical change detection via gradient (strong change, but no change back)
11. Probably good data (=short periods of good data between long periods of bad data)

In order to perform some of the quality check steps, filled timeseries without NaNs are needed. Thus, the following three filling appraoches are used:
1. Polynomial filled series
2. Polynomial fitting series based on polynomial filled series over 14 hours
3. Spline fitting over roughly 14 hours based on existing measurements

Upcoming steps to be included (potentially):
1. Supervised ML to flag missed faulty measurements
2. Deshifting of relevant periods
3. Interpolation of missing periods

# Execute the Python Scripts 
This repository is structured in unittest (saved under tests) and main scripts in the source folder. All outcomes are saved in the output folder, but never committed to Gitlab. In order to run the code, the unittests needs to be executed.

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
conda activate my_env
code .
```
3. Activate the correct environment (here: conda environment called my_env)

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

## License
???

## Project status
The preprocessing of the data through statistical tools and existing python scripts is done. Now, it is looked into how to flag undetected measurement errors using supervied machine learning.