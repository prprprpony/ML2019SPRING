#!/bin/bash
comm -2 -3 <(sort $1) <(sort $2) | wc -l
