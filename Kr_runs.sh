#!/bin/bash

for i in {0..49}
do
cd example_$i
sbatch job.sub
cd ..
done