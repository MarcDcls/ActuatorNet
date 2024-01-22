import numpy as np
from torch.utils.data import Dataset as TorchDataset

class Dataset(TorchDataset):
    """
    A generic dataset class compatible with PyTorch DataLoader.
    """
    def __init__(self, inputs_size: int, outputs_size: int):
        self.inputs = []
        self.outputs = []
        self.inputs_size = inputs_size
        self.outputs_size = outputs_size
        self.inputs_scales = []
        self.outputs_scales = []
        self.size = 0
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, index: int):
        return {"input": self.inputs[index], "output": self.outputs[index]}
    
    def __getitems__(self, indices: list):
        return [{"input": self.inputs[index], "output": self.outputs[index]} for index in indices]
    
    def add(self, input: list, output: list) -> None:
        """
        Add an entry to the dataset.
        """
        if len(input) != self.inputs_size:
            raise ValueError(f"Input size should be {self.inputs_size} but is {len(input)}")
        if len(output) != self.outputs_size:
            raise ValueError(f"Output size should be {self.outputs_size} but is {len(output)}")
        
        self.inputs.append(np.float32(input))
        self.outputs.append(np.float32(output))
        self.size += 1

    def shuffle(self) -> None:
        """
        Shuffle the dataset entries.
        """
        indices = np.arange(self.size)
        np.random.shuffle(indices)
        self.inputs = [self.inputs[index] for index in indices]
        self.outputs = [self.outputs[index] for index in indices]

    def split(self, ratio: float) -> ("Dataset", "Dataset"):
        """
        Split the dataset into two datasets of sizes ratio and 1 - ratio.
        """
        if ratio < 0 or ratio > 1:
            raise ValueError(f"Ratio should be between 0 and 1 but is {ratio}")
        
        dataset_1 = Dataset(self.inputs_size, self.outputs_size)
        dataset_2 = Dataset(self.inputs_size, self.outputs_size)

        indices = np.arange(self.size)
        np.random.shuffle(indices)
        split_index = int(self.size * ratio)
        indices_1 = indices[:split_index]
        indices_2 = indices[split_index:]

        dataset_1.inputs = [self.inputs[index] for index in indices_1]
        dataset_1.outputs = [self.outputs[index] for index in indices_1]
        dataset_1.size = len(indices_1)

        dataset_2.inputs = [self.inputs[index] for index in indices_2]
        dataset_2.outputs = [self.outputs[index] for index in indices_2]
        dataset_2.size = len(indices_2)

        return dataset_1, dataset_2

    def compute_scales(self, x_min: float = -1, x_max: float = 1, y_min: float = -1, y_max: float = 1) -> None:
        """
        Compute the scales used to rescale the input and output data. This method use the 
        Interquartile Range Method (IQR) to exclude outliers.
        """
        inputs = np.array(self.inputs)
        outputs = np.array(self.outputs)

        # Compute the IQR
        q1 = np.quantile(inputs, 0.25, axis=0)
        q3 = np.quantile(inputs, 0.75, axis=0)
        iqr = q3 - q1

        # Remove the ouliers
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = np.where((inputs < lower_bound) | (inputs > upper_bound))

        inputs = np.delete(inputs, outliers[0], axis=0)
        outputs = np.delete(outputs, outliers[0], axis=0)

        # Compute the scalers
        self.inputs_scales = np.float32((x_max - x_min) / (np.max(inputs, axis=0) - np.min(inputs, axis=0)))
        self.outputs_scales = np.float32((y_max - y_min) / (np.max(outputs, axis=0) - np.min(outputs, axis=0)))

    def save(self, filename: str) -> None:
        """
        Save the dataset in a compressed numpy file.
        """
        np.savez_compressed(filename, 
                            inputs=self.inputs, 
                            outputs=self.outputs, 
                            inputs_scales=self.inputs_scales, 
                            outputs_scales=self.outputs_scales)

    @classmethod
    def load(cls, filename: str) -> "Dataset":
        """
        Load the dataset from a compressed numpy file.
        """
        data = np.load(filename)
        inputs_size = len(data["inputs"][0])
        outputs_size = len(data["outputs"][0])

        dataset = cls(inputs_size, outputs_size)
        dataset.inputs = np.float32(data["inputs"])
        dataset.outputs = np.float32(data["outputs"])
        dataset.inputs_scales = np.float32(data["inputs_scales"])
        dataset.outputs_scales = np.float32(data["outputs_scales"])
        dataset.size = len(dataset.inputs)
        
        return dataset
    
