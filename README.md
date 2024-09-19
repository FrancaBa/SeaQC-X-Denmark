# Quality Check for Tide Gauge Measurements in Greenland

In the scope of the GrønSL project, this repository has been developed to have an automated quality check algorithm for sea level data in Greenland used on state of the art mchine learning approaches.

## Goal
The goal of this work is to generate an automated QC algorithm which can detect and adapt faulty measurements in timeseries using ML. The first version will be focusing on checking sea level measurements in Greenland.

## Motivation
Quality checking of many time series is at the moment only sparsly done and often manually. However, with the increasing amount of data, it is important to have an automated approach to assess the quality of measurements/recieved values - As a model can only be as accurate as the input. Thus, it is important to have cleaned input data especially in regions with little measurements as Greenland. This work will focus on developing an automated quality check algorithm for sea level data with a strong focus on detecting shifts and deshifting.

## Roadmap
This project will start off by developing a ML approach to flag tide gauge measurements in adequate groups. The next step will then be to adapt the flagged values to create a 'correct' ts. The timeline for this project is around 10 months.

# Pull and Push Code
The subsequent lines are describing how to pull and push changes to this repository.

## Pull code from Gitlab
1. Add a new branch on Gitlab via +-Icon
* Naming convension: initials + month abbr. + number edit in this month 
* f.e: fb_sep1, fb_sep2, ...
2. Pull code to local server (you will need to log in to your Gitlab account)
```
git clone --branch <branch_name> https://gitlab.dmi.dk/frb/qc_sl_greenland greenland-qc
conda activate my_env
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

# Contributing
Contributions to this project are welcome. Please fork the repository, make your changes, and submit a merge request with a clear description of your modifications.

## Authors and acknowledgment
This project is carried out by Franca Bauer under the supervision of Jian Su. By question, please feel free to reach out to frb@dmi.dk or jis@dmi.dk.

## License
???

## Project status
This is just the beginning!
