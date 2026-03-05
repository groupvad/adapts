import torch
import random

class Memory:
    def __init__(self, memory_size: int):
        self.memory_size = memory_size
        self.tasks_memory = {}
        self.tasks_seen = {}       
        self.num_tasks = 0

    def _rebalance(self):
        task_quota = self.memory_size // self.num_tasks

        for task_id in self.tasks_memory:
            # Randomly drop samples until we meet the new quota for every task memory
            while len(self.tasks_memory[task_id]) > task_quota:
                idx = random.randrange(len(self.tasks_memory[task_id]))
                self.tasks_memory[task_id].pop(idx)

    def add_samples(self, task_id: int, images: torch.Tensor, aug_images: torch.Tensor, masks: torch.Tensor):
        # Ensure the batches match in size
        assert len(images) == len(aug_images) == len(masks), "Batch sizes for images, aug_images, and masks must match."

        if task_id not in self.tasks_memory:
            self.tasks_memory[task_id] = []
            self.tasks_seen[task_id] = 0
            self.num_tasks += 1
            self._rebalance() # for making space for the new task

        task_quota = self.memory_size // self.num_tasks

        # Zip the three tensors together to iterate over the batch simultaneously
        for img, aug_img, mask in zip(images, aug_images, masks):
            self.tasks_seen[task_id] += 1

            # Store them as a tuple of cloned tensors
            sample_tuple = (img.clone(), aug_img.clone(), mask.clone())

            # Fill memory up to the quota
            if len(self.tasks_memory[task_id]) < task_quota:
                self.tasks_memory[task_id].append(sample_tuple)
            else:
                # Reservoir Sampling: Decreasing probability of overwrite over time
                j = random.randint(0, self.tasks_seen[task_id] - 1)
                if j < task_quota:
                    self.tasks_memory[task_id][j] = sample_tuple

    def get_samples(self, n_replay_samples: int):
        # 1. Determine how many samples to draw from each task
        if self.num_tasks == 0:
            return torch.empty(0), torch.empty(0), torch.empty(0)

        quotas = [n_replay_samples // self.num_tasks] * self.num_tasks
        remainder = n_replay_samples % self.num_tasks

        # Distribute the remainder randomly across tasks
        task_indices = list(range(self.num_tasks))
        random.shuffle(task_indices)
        for i in range(remainder):
            quotas[task_indices[i]] += 1

        # Lists to hold the separated samples
        ret_images = []
        ret_aug_images = []
        ret_masks = []

        task_ids = list(self.tasks_memory.keys())

        # 2. Extract the samples based on the calculated quotas
        for task_id, quota in zip(task_ids, quotas):
            memory_samples = self.tasks_memory[task_id]
            if not memory_samples:
                continue

            n_samples = min(quota, len(memory_samples))
            samples_idx = torch.randperm(len(memory_samples))[:n_samples]

            for idx in samples_idx:
                # Unpack the stored tuple
                img, aug_img, mask = memory_samples[idx]

                # Unsqueeze to add a batch dimension back before concatenating
                ret_images.append(img.unsqueeze(dim=0))
                ret_aug_images.append(aug_img.unsqueeze(dim=0))
                ret_masks.append(mask.unsqueeze(dim=0))

        if not ret_images:
             return torch.empty(0), torch.empty(0), torch.empty(0)

        # 3. Concatenate and return three separate batch tensors
        return torch.cat(ret_images), torch.cat(ret_aug_images), torch.cat(ret_masks)