#开发一个日志系统， 既要把日志输出到控制台， 还要写入日志文件  
import logging

__name__=''

def test():
    print('test')
    
class MyLogger():
    format_dict = {
        1 : logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        2 : logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        3 : logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        4 : logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        5 : logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    }

    def __init__(self, logname, loglevel, logger):
        '''
           指定保存日志的文件路径，日志级别，以及调用文件
           将日志存入到指定的文件中
        '''

        # 创建一个logger
        self.logger = logging.getLogger(logger)
        self.logger.setLevel(logging.DEBUG)

        # 创建一个handler，用于写入日志文件
        fh = logging.FileHandler(logname)
        fh.setLevel(logging.DEBUG)

        # 再创建一个handler，用于输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # 定义handler的输出格式
        formatter = logging.Formatter('%(asctime)s %(name)s %(thread)d %(levelname)s %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # only first time need to add the handler
        if(self.logger.hasHandlers() == False):
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

   
    def getlog(self):
        return self.logger


