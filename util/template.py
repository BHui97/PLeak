class TextTemplate:
    def __init__(self, prefix_1=None, prefix_2=None):
        self.prefix_1 = prefix_1
        self.prefix_2 = prefix_2
        self.prefix_trigger = '' if prefix_2 is None else self.prefix_1
        # self.prefix_trigger = ''

    def __call__(self, input_1=None, input_2=None):
        ret = self.prefix_1 + input_1
        if self.prefix_2 is None or input_2 is None:
            return ret + '\n'
        else:
            return ret + ' ' + self.prefix_2 + input_2 + '\n'
    
    def format_trigger(self, trigger):
        return self.prefix_trigger + trigger +'\n'
        # return self.prefix_trigger + '\'\'\n'+trigger + '\'\'\n'




