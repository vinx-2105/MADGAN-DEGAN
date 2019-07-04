class Logger:

    def __init__(self, filepath):
        self.filepath = filepath
        with open(self.filepath, 'w') as f:
            f.write("Logger")
    
    def log(self, log_string):
        with open(self.filepath, 'a') as f:
            f.writelines([log_string, '\n'])