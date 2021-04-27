This respository is for setting up and testing on our ec2 instances. If you are doing exploratory data science then you probably want 
to head over to the analytics repository where all this code is imported. The analytics repository has different setup 
instructions, but they both use the same `reactn` conda virtual environment.

## Quick Start
**Production ec2 details**:

```
ssh -i ~/.ssh/react.pem ubuntu@ec2-18-217-202-94.us-east-2.compute.amazonaws.com
cd reactrack
sudo /home/ubuntu/anaconda3/envs/reactn/bin/python app.py > log.log 2> error.log &
```
Process link: http://18.217.202.94/process
   
   
**Staging ec2 details**:

```
ssh -i ~/.ssh/react.pem ubuntu@ec2-18-222-34-139.us-east-2.compute.amazonaws.com
cd reactrack
sudo /home/ubuntu/anaconda3/envs/reactn/bin/python app.py > log.log 2> error.log &
```
Process link: http://18.222.34.139/process

## Documentation to launch new EC2 instance

Oh no! Something has gone wrong with the current API and it appears no amount of StackOverflow is going to fix it! 

In the event of this calamity, do not fret. A new, clean EC2 instance can be set up relatively painlessly. If you're
just trying to access an already existing API, please skip to the section labeled "API" below. If at any point you run into
difficulties, please check the bottom of this page to see if the issue has been caught before!


### Launching New EC2 Instance
 
1. Log into the AWS console using the [DataScience credentials](https://start.1password.com/open/i?a=SHYVNHPSWVA67JVYED7MX76CEA&h=reactneuro.1password.com&i=slww4ppewiijrcshaprsz435cq&v=qlabfski72xgk3zmeze7i7bsmi) 
(1password)
2. Under "Compute," select "EC2"
3. Click on "Launch Instance" and search for the following AMI:
    ```
    AMI: anaconda3-5.1.0-on-ubuntu-16.04-lts (ami-47d5e222)
    ```

4. Choose `t2.large` as the `instance type`
5. On the _Configure Instance Details_ page, change `Subnet` to `subnet-a85304d2` (should correspond to availability zone `us-east-2b`)
6. On the storage page, input 50 GiB
7.  On the _Configure Security Group_ page, Select an existing security group and enable 
    ```
    sg-049c42f0a1ea82701 launch-wizard-1
    ````
8. Once this is all complete, navigate to "Review and Launch"
9. When prompted, choose an existing key pair (`react`) and launch the instance.
10. View Instances to verify that the new instance has been created properly. It will need to initialize first, which will be indicated under "Status Checks" (should only take a minute or two)

Once this is all complete you can SSH into the instance using the command below
```
ssh -i ~/.ssh/react.pem ubuntu@ec2-3-22-175-44.us-east-2.compute.amazonaws.com
```
replacing the address following @ with the newly created address (can be found on the Instances Page).

### Setting up reactrack repositories
You can now set up the reactrack git repositories according to the Installation instructions below


## Installation of reactrack
1. **Clone reactrack repository onto instance**
    ```
    git clone https://github.com/react-neuro/reactrack
    cd reactrack
    ```

2. **Get the code in from the matviz submodule**
    ```
    git submodule update --init
    ```

3. **Get the following secret files and add to `reactrack/.aws`**

    - Download [credentials.dms](https://reactneuro.1password.com/vaults/qlabfski72xgk3zmeze7i7bsmi/allitems/ytx7asqecc2kpku5v3ayvvq6f4) 
    from 1password and rename to `credentials`. 
    - Download the [production_staging_config.json](https://reactneuro.1password.com/vaults/qlabfski72xgk3zmeze7i7bsmi/allitems/63adwbrkhar6brdai6fwa47wxy)
    - Download the [production_db.json](https://reactneuro.1password.com/vaults/qlabfski72xgk3zmeze7i7bsmi/tags/g7dtcfxjmxbn56vsqg43nmrypl/ndzsjh5glgaxrqlleuh6p67mie)
    - Download the [staging_db.json](https://reactneuro.1password.com/vaults/qlabfski72xgk3zmeze7i7bsmi/allitems/qpkc4iyq6vza7y2svf4g5ebdcm)
    - Download the [sentiment-ibm-credentials.json](https://reactneuro.1password.com/vaults/qlabfski72xgk3zmeze7i7bsmi/allitems/4sq2pqpn2ptwdhjfbmxcv5qdc4)
    - Download [speech-api-google-credentials.json](https://reactneuro.1password.com/vaults/qlabfski72xgk3zmeze7i7bsmi/allitems/3i6eytkcfnapagdw5guaikdbxe)
    
    All files should now be located in your local downloads folder.
    
    - Create hidden folder in `reactrack` (on the server) with following command:
        ```
        mkdir .aws
        ```

   - To add the above files to `reactrack`, use following commands **on your local machine**
        ```
        scp -i ~/.ssh/react.pem /local/path/to/credentials ubuntu@[ec2_address]:/home/ubuntu/reactrack/.aws
        scp -i ~/.ssh/react.pem /local/path/to/production_staging_config.json ubuntu@[ec2_address]:/home/ubuntu/reactrack/.aws
        scp -i ~/.ssh/react.pem /local/path/to/production_db.json ubuntu@[ec2_address]:/home/ubuntu/reactrack/.aws
        scp -i ~/.ssh/react.pem /local/path/to/staging_db.json ubuntu@[ec2_address]:/home/ubuntu/reactrack/.aws
        scp -i ~/.ssh/react.pem /local/path/to/speech-api-google-credentials.json ubuntu@[ec2_address]:/home/ubuntu/reactrack/.aws
        scp -i ~/.ssh/react.pem /local/path/to/sentiment-ibm-credentials.json ubuntu@[ec2_address]:/home/ubuntu/reactrack/.aws
        ```
        replacing `[ec2_address]` with the Public DNS of the new instance you just created.


4. **Setup virtual environment**
    ```
    conda env create -f environment.yml
    source activate reactn
    ```

5. **Install `librosa` and `ffmpeg` for audio and video**
    ```
   conda install -c conda-forge librosa
   ```
   ```
   sudo apt-get update
   sudo apt-get install ffmpeg 
    ```
6. Install `en_core_web_sm` for spacy
    ```
    python -m spacy download en_core_web_sm
    ```



## API
We have a flask API to run on EC2 that will proccess data triggered when CSVs are uploaded. 
**The following instructions are all for the staging ec2, but the same procedure applies for the production ec2** 

### Set-Up
**To set up your system for interaction with the EC2 instance, please do the following:**

1. Create a new top-level hidden folder `.aws` in your local `reactrack` and put the following files inside:
[AWS Credentials file](https://reactneuro.1password.com/vaults/qlabfski72xgk3zmeze7i7bsmi/allitems/ytx7asqecc2kpku5v3ayvvq6f4) 
inside (if you cannot create a folder with a name preceded by `.`, first show all hidden files by using `Cmd + Shift + .`)
2. Download the following [AWS Key](https://reactneuro.1password.com/vaults/qlabfski72xgk3zmeze7i7bsmi/allitems/au5udilcpreujnkjnfeqrxl6xm) from 1password
and place inside your `~/.ssh` directory
3. run `chmod 400 ~/.ssh/react.pem` to grant the proper permissions. [Reference link to command](https://unix.stackexchange.com/questions/115838/what-is-the-right-file-permission-for-a-pem-file-to-ssh-and-scp)

You should now be all set to use the server from here on out!

To test that everything works, run the following:

```
ssh -i ~/.ssh/react.pem ubuntu@ec2-18-222-34-139.us-east-2.compute.amazonaws.com
```
then:
```
cd reactrack
tail -f error.log
```
This will print out the error log of any currently running processes. The commands in the next section are to help 
you actually run the API when trying to make changes. 



### Running the api
```
ssh -i ~/.ssh/react.pem ubuntu@ec2-18-222-34-139.us-east-2.compute.amazonaws.com
cd reactrack & sudo /home/ubuntu/anaconda3/envs/reactn/bin/python app.py > log.log 2> error.log &
```

Then to kick the API off you need to call this link:
http://18.222.34.139/process

To re-run a specific exam chart after doing a pull, just pass in the instructions as a json string:
```
sudo /home/ubuntu/anaconda3/envs/reactn/bin/python app.py '{"videos": false, "host": "aws", "exams": ["Prosaccade", "Convergence"]}' > log.log 2> error.log & tail -f error.log 
```

example parameters:
```
params = {"videos": false, 
          "host": "local aws",
          "exams": ["Prosaccade", "Convergence"],
          "control_subj_quantity": 30,
          "overwrite": True # whether the data processing should be redone
          }

```

* exams: can be a list of exams, or a string with a single exam, case insensitive
* overwrite:
  * True-> redo the analysis no matter what
  * False-> redo the analysis only if a file is missing
  * Integer-> redo the analysis if that many hours passed since the last processing




Checking if the API is running, and killing it if you need to
```
ps -ef | grep python
ps -ef | grep "python" | awk '{print $2}' | xargs sudo kill
```

Checking in on the log file as it runs:
```
tail -f error.log
```

## updating the API
To update reactrack code on the ec2 instances, use `update_reactrack.sh`
To run:

**ssh into the server:**
```
ssh -i ~/.ssh/react.pem ubuntu@ec2-18-222-34-139.us-east-2.compute.amazonaws.com
```

**Run update_reactrack.sh:**
```
cd reactrack
bash update_reactrack.sh
```

`update_reactrack.sh` contains the following:

```
# pulls latest changes from git
git pull 

# update submodules
git submodule update

# kills all existing processes
ps -ef | grep "python" | awk '{print $2}' | xargs sudo kill

# start up app.py again
sudo /home/ubuntu/anaconda3/envs/reactn/bin/python app.py > log.log 2> error.log &
```

Go to the following link to start it processing:
http://18.222.34.139/process


## testing

Go to the `reactrack` directory and run this line to test:

```python -m pytest test.py```


## Places where the code is brittle

Below are a few cases where the code is brittle, things to look out for:

Question information is loaded from three types:
 - columns in the users table
 - questions answered once per user, but are in the question_responses table
 - questions answered once per exam in the question_responses table

The short code for the exam questions is hard-coded. If we add questions we'll need to add them in to utils_db.py
an alternative is to add a 'shorthand' to the database
```
tmp = {
        1 :"eyecolor",
        2 :"concussion",
        3 :"imparment",
        4 :"examcomments",
        5 :"glasses",
        6 :"administrator",
        7 :"generalcomments",
        8: "q_gender",
        9: "q_email"
      }
```
Gender and email are both in the questions table, and stored duplicatedly in the users table. They are occasionally being answered in the question_responses, and in those cases they are manually removed... by NUMBER, so if we remove them and start using questions ID 8 or 9 then we'll need to do something about it.

If an exam has no columns associated with it - then it is skipped. But if it has not all of the exam columns in there - for example it is partially uploaded - then the whole data will be processed.
If we want to account for that we'll need to have some knowledge about which columns are supposed to go with which exams.

## legacy code
Code has been removed from the current master with the following legacy capabilities:
 - load dataframes from CSV robustly handling column name formats and other minutia
 - generate and process UUIDs and Json files related to the original ipad app integration and file-based data collections.
This code can be found back in the following commit:
reactrack => `commit e8226f40df483f59f7b9a450a6a8e4356f8e2209`
and in a more complete state in the analytics repository before reactrack was separated:
analytics => `commit 4c27bf76aa35aaeaf1859eff4355242429804d90`


## misc. tips and tricks
A few of the issues that I came across:
* if your conda env fails to install due to `insufficient space`, this is likely because you may have forgotten to set the
storage amount to 50GiB. Unfortunately, there's no way to increase this limit after creating an instance :( so you'll need 
to make another one
    - To check how much storage you have on this instance, use command `df -h` on the command line
    - You should see something like 
        ```
      /dev/xvda1       49G  8.8G   40G
      ```
      If the first number says 8G, then you may not have changed the default and will need to create a new instance.
* When trying to install the `unzip` package (or any others), you may see the following error:
    ```
    E: Could not get lock /var/lib/dpkg/lock – open (11: Resource temporarily unavailable)
    E: Unable to lock the administration directory (/var/lib/dpkg/), is another process using it?
    ```
  This may be because you have another instance running in another terminal window or tab. I'm not exactly sure why this
  causes problems, but it fixed itself for me when I logged out of the conflicting instance.
  
  If this does not fix it, another process may be using it (as the message indicates) so wait a few minutes and try again. 