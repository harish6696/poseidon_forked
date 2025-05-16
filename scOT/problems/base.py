"""
This file contains the dataset selector get_dataset, as well as the base 
classes for all datasets.
"""

from torch.utils.data import Dataset, ConcatDataset
from typing import Optional, List, Dict
from abc import ABC
import re
import os
import shutil
from accelerate.utils import broadcast_object_list


def get_dataset(dataset, **kwargs):
    """
    Get a dataset by name.
    If you enter a list of str, will return a ConcatDataset of the datasets.

    Available choices are:
    - fluids.incompressible.BrownianBridge(.tracer)
    - fluids.incompressible.Gaussians(.tracer)
    - fluids.incompressible.ShearLayer
    - fluids.incompressible.Sines(.tracer)
    - fluids.incompressible.PiecewiseConstants(.tracer)
    - fluids.incompressible.VortexSheet(.tracer)
    - fluids.incompressible.forcing.KolmogorovFlow
    - fluids.compressible.gravity.RayleighTaylor(.tracer)
    - fluids.compressible.RiemannKelvinHelmholtz
    - fluids.compressible.RiemannCurved
    - fluids.compressible.Riemann
    - fluids.compressible.KelvinHelmholtz
    - fluids.compressible.Gaussians
    - fluids.compressible.RichtmyerMeshkov(.tracer)
    - fluids.compressible.steady.Airfoil(.time)
    - elliptic.poisson.Gaussians(.time)
    - elliptic.Helmholtz(.time)
    - wave.Layer
    - wave.Gaussians
    - reaction_diffusion.AllenCahn

    Adding .out at the end of the str, returns a dataset with more time steps.
    **kwargs overwrite the default settings.
    .time is a time-wrapped time-independent dataset.
    """
    if isinstance(dataset, list):  #dataset example for PT: fluids.compressible.Riemann
        return ConcatDataset([get_dataset(d, **kwargs) for d in dataset])
    if "fluids" in dataset:
        if "fluids.incompressible" in dataset:
            if "BrownianBridge" in dataset:
                from .fluids.incompressible import BrownianBridge as dset
            elif "Gaussians" in dataset:
                from .fluids.incompressible import Gaussians as dset
            elif "ShearLayer" in dataset:
                from .fluids.incompressible import ShearLayer as dset
            elif "Sines" in dataset:
                from .fluids.incompressible import Sines as dset
            elif "PiecewiseConstants" in dataset:
                from .fluids.incompressible import PiecewiseConstants as dset
            elif "VortexSheet" in dataset:
                from .fluids.incompressible import VortexSheet as dset
            elif "forcing" in dataset:
                if "KolmogorovFlow" in dataset:
                    from .fluids.incompressible import KolmogorovFlow as dset
                else:
                    raise ValueError(f"Unknown dataset {dataset}")
            else:
                raise ValueError(f"Unknown dataset {dataset}")
        
        elif "fluids.compressible" in dataset:
            if "gravity" in dataset:
                if "RayleighTaylor" in dataset:
                    from .fluids.compressible import RayleighTaylor as dset

                    if "out" in dataset:
                        default_time_settings = {
                            "max_num_time_steps": 10,
                            "time_step_size": 1,
                        }
                    else:
                        default_time_settings = {
                            "max_num_time_steps": 7,
                            "time_step_size": 1,
                        }
                    kwargs = {**default_time_settings, **kwargs}
                elif "Blast" in dataset:
                    from .fluids.compressible import Blast as dset
            elif "RiemannKelvinHelmholtz" in dataset:
                from .fluids.compressible import RiemannKelvinHelmholtz as dset
            elif "RiemannCurved" in dataset:
                from .fluids.compressible import RiemannCurved as dset
            elif "Riemann" in dataset:
                from .fluids.compressible import Riemann as dset
            elif "KelvinHelmholtz" in dataset:
                from .fluids.compressible import KelvinHelmholtz as dset
            elif "Gaussians" in dataset:
                from .fluids.compressible import Gaussians as dset
            elif "RichtmyerMeshkov" in dataset:
                from .fluids.compressible import RichtmyerMeshkov as dset
            elif "steady" in dataset:
                if "steady.Airfoil" in dataset:
                    from .fluids.compressible import Airfoil as dset

                    if "out" in dataset:
                        raise ValueError(f"Unknown dataset {dataset}")
                else:
                    raise ValueError(f"Unknown dataset {dataset}")
            else:
                raise ValueError(f"Unknown dataset {dataset}")
        
        else:
            raise ValueError(f"Unknown dataset {dataset}")
        if "out" in dataset: #this is during out of distribution inference
            default_time_settings = {"max_num_time_steps": 10, "time_step_size": 2}
        else:
            default_time_settings = {"max_num_time_steps": 7, "time_step_size": 2} #this is used in CERP for training/val and testing (inference)
        if "tracer" in dataset:
            tracer = True
        else:
            tracer = False
        if not "steady" in dataset:
            kwargs = {"tracer": tracer, **default_time_settings, **kwargs}
    elif "elliptic" in dataset:
        if ".out" in dataset:
            raise NotImplementedError(f"Unknown dataset {dataset}")
        if "elliptic.poisson" in dataset:
            if "Gaussians" in dataset:
                from .elliptic.poisson import Gaussians as dset
            else:
                raise ValueError(f"Unknown dataset {dataset}")
        elif "elliptic.Helmholtz" in dataset:
            from .elliptic.helmholtz import Helmholtz as dset
        else:
            raise ValueError(f"Unknown dataset {dataset}")
    elif "wave" in dataset:
        if "wave.Layer" in dataset:
            if "out" in dataset:
                default_time_settings = {"max_num_time_steps": 10, "time_step_size": 2}
            else:
                default_time_settings = {"max_num_time_steps": 7, "time_step_size": 2}
            kwargs = {**default_time_settings, **kwargs}
            from .wave.acoustic import Layer as dset
        elif "wave.Gaussians" in dataset:
            if "out" in dataset:
                raise ValueError(f"Unknown dataset {dataset}")
            else:
                default_time_settings = {"max_num_time_steps": 7, "time_step_size": 2}
            kwargs = {**default_time_settings, **kwargs}
            from .wave.acoustic import Gaussians as dset
        else:
            raise ValueError(f"Unknown dataset {dataset}")
    elif "reaction_diffusion" in dataset:
        if "reaction_diffusion.AllenCahn" in dataset:
            if "out" in dataset:
                default_time_settings = {"max_num_time_steps": 9, "time_step_size": 2}
            else:
                default_time_settings = {"max_num_time_steps": 7, "time_step_size": 2}
            kwargs = {**default_time_settings, **kwargs}
            from .reaction_diffusion.allen_cahn import AllenCahn as dset
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    return dset(**kwargs) if ".time" not in dataset else TimeWrapper(dset(**kwargs))

#for time "in"dependent problems use BaseDataset
class BaseDataset(Dataset, ABC):
    """A base class for all datasets. Can be directly derived from if you have a steady/non-time dependent problem."""

    def __init__(
        self,
        which: Optional[str] = None,
        num_trajectories: Optional[int] = None,
        data_path: Optional[str] = "./data",
        move_to_local_scratch: Optional[str] = None,
    ) -> None:
        """
        Args:
            which: Which dataset to use, i.e. train, val, or test.
            resolution: The resolution of the dataset.
            num_trajectories: The number of trajectories to use for training.
            data_path: The path to the data files.
            move_to_local_scratch: If not None, move the data to this directory at dataset initialization and use it from there.
        """
        assert which in ["train", "val", "test"]
        assert num_trajectories is not None and (
            num_trajectories > 0 or num_trajectories in [-1, -2, -8]
        )

        self.num_trajectories = num_trajectories
        self.data_path = data_path
        self.which = which
        self.move_to_local_scratch = move_to_local_scratch

    def _move_to_local_scratch(self, file_path):
        if self.move_to_local_scratch is not None:  #by default it is None
            data_dir = os.path.join(self.data_path, file_path)
            file = file_path.split("/")[-1]
            scratch_dir = self.move_to_local_scratch
            dest_dir = os.path.join(scratch_dir, file)
            RANK = int(os.environ.get("LOCAL_RANK", -1))
            if not os.path.exists(dest_dir) and (RANK == 0 or RANK == -1):
                print(f"Start copying {file} to {dest_dir}...")
                shutil.copy(data_dir, dest_dir)
                print("Finished data copy.")
            # idk how to do the barrier differently
            ls = broadcast_object_list([dest_dir], from_process=0)
            dest_dir = ls[0]
            return dest_dir
        else:
            return file_path

    def post_init(self) -> None:
        """
        Call after self.N_max, self.N_val, self.N_test, as well as the file_paths and normalization constants are set.
        """
        assert (
            self.N_max is not None
            and self.N_max > 0
            and self.N_max >= self.N_val + self.N_test
        )
        if self.num_trajectories == -1: #consider all data available for training
            self.num_trajectories = self.N_max - self.N_val - self.N_test 
        elif self.num_trajectories == -2: #consider only half of the data for training
            self.num_trajectories = (self.N_max - self.N_val - self.N_test) // 2
        elif self.num_trajectories == -8: #consider only one eigth of the data for training
            self.num_trajectories = (self.N_max - self.N_val - self.N_test) // 8
        assert self.num_trajectories + self.N_val + self.N_test <= self.N_max
        assert self.N_val is not None and self.N_val > 0
        assert self.N_test is not None and self.N_test > 0
        # the below setup is for the time-independent problems (different for the time-dependent problems: BaseTimeDataset)
        if self.which == "train":
            self.length = self.num_trajectories
            self.start = 0
        elif self.which == "val":
            self.length = self.N_val
            self.start = self.N_max - self.N_val - self.N_test
        else:
            self.length = self.N_test
            self.start = self.N_max - self.N_test

        self.output_dim = self.label_description.count(",") + 1
        descriptors, channel_slice_list = self.get_channel_lists(self.label_description)
        self.printable_channel_description = descriptors
        self.channel_slice_list = channel_slice_list

    def __len__(self) -> int:
        """
        Returns: overall length of dataset.
        """
        return self.length

    def __getitem__(self, idx) -> Dict:
        """
        Get an item. OVERWRITE!

        Args:
            idx: The index of the sample to get.

        Returns:
            A dict of key-value pairs of data.
        """
        pass

    @staticmethod
    def get_channel_lists(label_description):
        matches = re.findall(r"\[([^\[\]]+)\]", label_description)
        channel_slice_list = [0]  # use as channel_slice_list[i]:channel_slice_list[i+1]
        beautiful_descriptors = []
        for match in matches:
            channel_slice_list.append(channel_slice_list[-1] + 1 + match.count(","))
            splt = match.split(",")
            if len(splt) > 1:
                beautiful_descriptors.append("".join(splt))
            else:
                beautiful_descriptors.append(match)
        return beautiful_descriptors, channel_slice_list

# for time dependent problems use BaseTimeDataset
class BaseTimeDataset(BaseDataset, ABC):
    """A base class for time dependent problems. Inherit time-dependent problems from here."""

    def __init__(
        self,
        *args,
        max_num_time_steps: Optional[int] = None,
        time_step_size: Optional[int] = None,
        fix_input_to_time_step: Optional[int] = None,
        allowed_time_transitions: Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        """
        Args:
            max_num_time_steps: The maximum number of time steps to use.
            time_step_size: The size of the time step.
            fix_input_to_time_step: If not None, fix the input to this time step.
            allowed_time_transitions: If not None, only allow these time transitions (time steps).
        """
        assert max_num_time_steps is not None and max_num_time_steps > 0
        assert time_step_size is not None and time_step_size > 0
        assert fix_input_to_time_step is None or fix_input_to_time_step >= 0

        super().__init__(*args, **kwargs)
        self.max_num_time_steps = max_num_time_steps
        self.time_step_size = time_step_size
        self.fix_input_to_time_step = fix_input_to_time_step
        self.allowed_time_transitions = allowed_time_transitions

    def _idx_map(self, idx):
        i = idx // self.multiplier
        _idx = idx - i * self.multiplier

        if self.fix_input_to_time_step is None:
            t1, t2 = self.time_indices[_idx] #self.time_indices are the same  for each trajectory
            assert t2 >= t1
            t = t2 - t1
        else:
            t1 = self.fix_input_to_time_step
            t2 = self.time_step_size * (_idx + 1) + self.fix_input_to_time_step
            t = t2 - t1
        return i, t, t1, t2

    def post_init(self) -> None:
        """
        Call after self.N_max, self.N_val, self.N_test, as well as the file_paths and normalization constants are set.
        self.max_time_step must have already been set.
        """
        assert (
            self.N_max is not None
            and self.N_max > 0
            and self.N_max >= self.N_val + self.N_test
        ) 
        # for CERP: self.N_max=10,000 (Max trajectories available in the dataset) : 
        # self.N_val=120 (number of trajectories used for training), 
        # self.N_test (number of trajectories used for testing/inference)=240 as mentioned in class CompressibleBase()
        if self.num_trajectories == -1: #consider all data available for training, for CE-RP: self.N_max=10,000 : self.N_val=120, self.N_test=240 as mentioned in class CompressibleBase()
            self.num_trajectories = self.N_max - self.N_val - self.N_test #for CERP: 10,000-120-240=9640 training trajectories
        elif self.num_trajectories == -2: #consider only half of the data for training
            self.num_trajectories = (self.N_max - self.N_val - self.N_test) // 2
        elif self.num_trajectories == -8: #consider only one eigth of the data for training
            self.num_trajectories = (self.N_max - self.N_val - self.N_test) // 8
        assert self.num_trajectories + self.N_val + self.N_test <= self.N_max
        assert self.N_val is not None and self.N_val > 0
        assert self.N_test is not None and self.N_test > 0
        assert self.max_num_time_steps is not None and self.max_num_time_steps > 0
        #self.max_num_time_steps=7
        if self.fix_input_to_time_step is not None:
            self.multiplier = self.max_num_time_steps
        else: #self.time_indices required for all2all training
            self.time_indices = [] #self.time_step_size=2, self.max_num_time_steps=7 (for training/val), 
                                   #self.max_num_time_steps=1 (for testing)
            for i in range(self.max_num_time_steps + 1):
                for j in range(i, self.max_num_time_steps + 1):
                    if (
                        self.allowed_time_transitions is not None #none for training/val, [1] for testing
                        and (j - i) not in self.allowed_time_transitions #self.allowed_time_transition dictates which (i,j) pairs are allowed. if j-i doesnt exist in the list, then skip that pair and continue
                    ): #this is false while train/val (allowed_time_transitions=None); while testing, this is true
                        continue #comtinues to the j-loop (obv)
                    
                    self.time_indices.append( #if (j-i) is in the allowed_time_transitions list, then append the pair to the self.time_indices list with the appropriate time_step_size (i.e time skip)
                        (self.time_step_size * i, self.time_step_size * j)
                    )
            self.multiplier = len(self.time_indices)

            #for testing, the above loop runs with i=0,1 and j=0,1
                #for i=0, j=0, (0,0) is not added to the list as (j-i = 0) is not in the allowed_time_transitions list = [1]
                #for i=0, j=1, (0,1) is added to the list as (j-i = 1) is in the allowed_time_transitions list = [1]
                #self.time_indices = [(0,14)]# for one step in-distribution prediction
                #for i=1, j=1, (1,1) is not added to the list as (j-i = 0) is not in the allowed_time_transitions list = [1]
                #thats the end of the loop
                #self.time_indices = [(0,14)]# for one step in-distribution prediction

        #for CERP training/val, the above loop runs with i=0,1,2,3,4,5,6,7 and j=0,1,2,3,4,5,6,7
        #self.multiplier= 36 = len(self.time_indices)
        #self.time_indices for CERP
        #[(0, 0), (0, 2), (0, 4), (0, 6), (0, 8), (0, 10), (0, 12), (0, 14), 
        # (2, 2), (2, 4), (2, 6), (2, 8), (2, 10), (2, 12), (2, 14), 
        # (4, 4), (4, 6), (4, 8), (4, 10), (4, 12), (4, 14), 
        # (6, 6), (6, 8), (6, 10), (6, 12), (6, 14), 
        # (8, 8), (8, 10), (8, 12), (8, 14), 
        # (10, 10), (10, 12), (10, 14), 
        # (12, 12), (12, 14), 
        # (14, 14)]
        #the below setup is for the time-dependent problems (different for the time-independent problems in class BaseDataset) 
        if self.which == "train": #self.length is the number of ip-op pairs
            self.length = self.num_trajectories * self.multiplier #4608 =128*36 for CERP  (128 set in yaml)
            self.start = 0
        elif self.which == "val":
            self.length = self.N_val * self.multiplier
            self.start = self.N_max - self.N_val - self.N_test
        else:
            self.length = self.N_test * self.multiplier
            self.start = self.N_max - self.N_test
        #self.label_description='[rho],[u,v],[p]'
        self.output_dim = self.label_description.count(",") + 1
        descriptors, channel_slice_list = self.get_channel_lists(self.label_description)
        self.printable_channel_description = descriptors
        self.channel_slice_list = channel_slice_list


class TimeWrapper(BaseTimeDataset):
    """For time-independent problems to be plugged into time-dependent models."""
    def __init__(self, dataset):
        super().__init__(
            dataset.which,
            dataset.num_trajectories,
            dataset.data_path,
            None,
            max_num_time_steps=1,
            time_step_size=1,
        )
        self.dataset = dataset
        self.resolution = dataset.resolution
        self.input_dim = dataset.input_dim
        self.output_dim = dataset.output_dim
        self.channel_slice_list = dataset.channel_slice_list
        self.printable_channel_description = dataset.printable_channel_description

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {**self.dataset[idx], "time": 1.0}
