import Logger
import sys, os

def getLogger():
    # get the file name
    fileUrl = sys.argv[0]
    filepath, tmpfilename = os.path.split(fileUrl)
    shotname, extension = os.path.splitext(tmpfilename)
    
    # get the logger object
    logger = Logger.MyLogger(logname='log.txt', loglevel=1, logger=shotname).getlog()
    return logger