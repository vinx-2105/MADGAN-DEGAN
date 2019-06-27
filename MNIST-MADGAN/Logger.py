class Logger():

    def __init__(self, filepath):
        self.filepath = filepath
    
    def log(log_string):
        with open(self.filepath, 'a') as f:
            f.writeline(log_string)