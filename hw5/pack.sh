#!/bin/bash
o=$(pwd)
cd $1
tar -zcvf $o/$2.tgz *.png
