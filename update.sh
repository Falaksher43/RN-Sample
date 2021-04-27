#!/usr/bin/env bash
git submodule update --recursive
conda env update --name reactn --file environment.yml
