#!/bin/bash
o=$(pwd)
cd $1
tar -zcvf $o/$1.tgz *.png
