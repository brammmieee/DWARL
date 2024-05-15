#!/bin/bash
ps aux | grep webots | grep -v grep | awk '{print $2}' | while read pid; do
    echo "Killing process $pid related to webots"
    kill $pid  # or use kill -9 $pid if necessary
done
