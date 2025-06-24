#!/bin/bash
cd /home/site/wwwroot
npm install -g serve
serve -s build -l 8080
