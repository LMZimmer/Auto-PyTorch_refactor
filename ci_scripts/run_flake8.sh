#!/bin/bash
flake8 --application-import-names=autoPyTorch --max-line-length=125 --show-source --ignore W605,E402,W503 autoPyTorch test examples
