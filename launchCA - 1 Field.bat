@echo off

@echo ----------------------------------------
@echo Lancio Content Analyzer per description
@echo ----------------------------------------
python runCA.py description 1M

@echo ----------------------------------------
@echo Lancio Content Analyzer per tags
@echo ----------------------------------------
python runCA.py tags 1M

@echo ----------------------------------------
@echo Lancio Content Analyzer per genres
@echo ----------------------------------------
python runCA.py genres 1M

@echo ----------------------------------------
@echo Lancio Content Analyzer per reviews
@echo ----------------------------------------
python runCA.py reviews 1M