import torch
import jax

jax.distributed.initialize()

num_dev = jax.device_count()

print("Hello from node ", jax.process_index())

print("Total dev number: ", num_dev)


