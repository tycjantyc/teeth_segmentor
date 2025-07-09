from processing.utils import load_itk
import ast

class DataSize_Factory():
    def __init__(self, file_path = 'datasets/dataset_sizes.txt', dataset_name = 'agh'):
        
        self.file_path = file_path
        self.dataset_name = dataset_name

        self.max_size = None
        
        with open(file_path) as f:
            for line in f.readlines():
                if line.startswith(self.dataset_name):
                    self.max_size = ast.literal_eval(line.split(' - ')[1])   


    def calucalte_size(self, list_of_files):

        if self.max_size is not None:
            return self.max_size
        
        self.max_size = [1000, 1000, 1000]

        for file in list_of_files:
            volume = load_itk(file)
            sh = volume.shape

            for i in range(3):
                if sh[i] < self.max_size[i]:
                    self.max_size[i] = sh[i]

        with open(self.file_path, "a") as f:
            f.write(f"{self.dataset_name} - {self.max_size}")

        return self.max_size