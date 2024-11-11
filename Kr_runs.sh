#!/bin/bash

for i in {0..49}
do
cd run$i
sbatch job.sub
cd ..
done