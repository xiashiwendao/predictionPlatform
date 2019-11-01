# -*- coding: utf-8 -*-
from Logger import MyLogger
import logging
import sys, os

def getLogger():
    # get the file name
    fileUrl = sys.argv[0]
    filepath, tmpfilename = os.path.split(fileUrl)
    shotname, extension = os.path.splitext(tmpfilename)
    
    # get the logger object
    logger = MyLogger(logname='log.txt', loglevel=1, logger=shotname).getlog()
    return logger

logger =getLogger()
logger.debug("Hoory, show!")

