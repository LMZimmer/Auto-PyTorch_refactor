echo Testing revision $(git rev-parse HEAD) ...
echo Testing from directory `pwd`
conda list
make test
