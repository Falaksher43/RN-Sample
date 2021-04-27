# pulls latest changes from git
git pull 

# update submodules
git submodule update

# activates environment and updates to include new packages
source activate reactn

# kills all existing processes
ps -ef | grep "python" | awk '{print $2}' | xargs sudo kill

# start up app.py again
sudo /home/ubuntu/anaconda3/envs/reactn/bin/python app.py > log.log 2> error.log &
