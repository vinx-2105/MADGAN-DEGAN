class Logger:

    def __init__(self, filepath, mode='w'):
        self.filepath = filepath
        with open(self.filepath, mode) as f:
            f.write("Logger")
    
    def log(self, log_string):
        with open(self.filepath, 'a') as f:
            f.writelines([log_string, '\n'])