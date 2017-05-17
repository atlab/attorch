class Dataset:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __repr__(self):
        s = ['Inputs:']
        for k, v in self.inputs.items():
            s.append('\t{}:\t{}'.format(k, ' x '.join(map(str, v.shape))))

        s = ['Outputs:']
        for k, v in self.outputs.items():
            s.append('\t{}:\t{}'.format(k, ' x '.join(map(str, v.shape))))
        return '\n'.join(s)
