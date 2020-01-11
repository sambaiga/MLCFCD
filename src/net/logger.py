import logging
import time


def initilize_log(log_path):
	log_file_name = '{0}{1}.log'.format(log_path, time.strftime("%Y-%m-%d-%H:%M:%S").replace(':','-'))
	with open(log_file_name, 'w'):
		pass

	logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
	rootLogger = logging.getLogger()
	rootLogger.setLevel(logging.DEBUG)
	fileHandler = logging.FileHandler("{0}".format(log_file_name))
	fileHandler.setFormatter(logFormatter)
	rootLogger.addHandler(fileHandler)

	consoleHandler = logging.StreamHandler()
	consoleHandler.setFormatter(logFormatter)
	rootLogger.addHandler(consoleHandler)

	return rootLogger


def log(string, level='info'):

	if level == 'info':
	rootLogger.info(string)
	elif level == 'debug':
	rootLogger.debug(string)
	elif level == 'warning':
	rootLogger.warning(string)

