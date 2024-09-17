# Quality Check for Tide Gauge Measurements in Greenland


## Introduction
Choose a self-explaining name for your project.

## Goal
The goal of this work is to generate an automated QC algorithm which can detect and adapt faulty measurements using ML.

## Motivation
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Roadmap
This project will start off by developing a ML approach to flag tide gauge measurements in adequate groups. The next step will then be to adapt the flagged values to create a 'correct' ts. The timeline for this project is around 10 months.

# Pull and Push Code
## Pull code from Gitlab
```
git clone https://gitlab.dmi.dk/frb/qc_sl_greenland
cd qc_sl_greenland
git checkout -b <branch_name>
git branch
conda activate my_env
```

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
Open the gitlab repository online and manually merge the branch into main. Be aware of potential conflicts.

# Contributing
Contributions to this project are welcome. Please fork the repository, make your changes, and submit a merge request with a clear description of your modifications.

## Authors and acknowledgment
This project is carried out by Franca Bauer under the supervision of Jian Su. By question, please feel free to reach out to frb@dmi.dk or jis@dmi.dk.

## License
???

## Project status
This is just the beginning!
