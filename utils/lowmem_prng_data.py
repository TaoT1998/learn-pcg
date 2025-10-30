"""
Memory-efficient (lowmem) PRNG data generation utilities.

This module provides on-demand sequence generation that only stores PRNG parameters (a, c, x_0)
initially and generates full sequences during training. This significantly reduces
memory usage for large datasets while maintaining identical training behavior.
"""

import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
from typing import List, Dict, Tuple, Optional
from utils.prng_data import (
    find_as, find_coprimes,
    base_b_lcg, base_tlcg, base_b_pcg_rs, base_b_pcg_rr, base_b_pcg_xsh_rr,
    base_b_pcg_xsh_rs, base_b_pcg_xsl_rr, convert_to_base_b
)


# Validation functions
def validate_pcg_rs_constraints(m: int, control_bits: int, bits_to_keep: int) -> tuple[bool, str]:
    if control_bits is None or bits_to_keep is None:
        return False, "control_bits and bits_to_keep must not be None"
    bit_length = int(np.ceil(np.log2(m)))
    required = (2 ** control_bits - 1) + bits_to_keep
    if bit_length <= required:
        return False, f"PCG_RS constraint violated: bit_length ({bit_length}) must be > {required}"
    return True, ""


def validate_pcg_rr_constraints(m: int, control_bits: int, bits_to_keep: int) -> tuple[bool, str]:
    if control_bits is None or bits_to_keep is None:
        return False, "control_bits and bits_to_keep must not be None"
    required = 2 ** control_bits
    if bits_to_keep < required:
        return False, f"PCG_RR constraint violated: bits_to_keep ({bits_to_keep}) must be >= {required}"
    return True, ""


def validate_pcg_xsh_rr_constraints(m: int, control_bits: int, bits_to_keep: int) -> tuple[bool, str]:
    if control_bits is None or bits_to_keep is None:
        return False, "control_bits and bits_to_keep must not be None"
    bit_length = int(np.ceil(np.log2(m)))
    required = control_bits + bits_to_keep
    if bit_length <= required:
        return False, f"PCG_XSH_RR constraint violated: bit_length ({bit_length}) must be > {required}"
    return True, ""


def validate_pcg_xsh_rs_constraints(control_bits: int, bits_to_keep: int) -> tuple[bool, str]:
    if control_bits is None or bits_to_keep is None:
        return False, "control_bits and bits_to_keep must not be None"
    constant_shift = bits_to_keep - control_bits - 2 ** control_bits + 1
    if constant_shift <= 0:
        return False, f"PCG_XSH_RS constraint violated: constant_shift ({constant_shift}) must be > 0"
    return True, ""


def validate_prng_parameters(prng_type: str, **kwargs) -> tuple[bool, str]:
    """
    Validate parameters for a specific PRNG type.
    Returns (is_valid, error_message)
    """
    m = kwargs.get('m')
    control_bits = kwargs.get('control_bits')
    bits_to_keep = kwargs.get('bits_to_keep')
    
    # Check that m is not None - this is required for all PRNG types
    if m is None:
        return False, "Parameter 'm' must not be None"
    
    # Special validation for LCG: bits_to_keep should equal the bit length of m
    if prng_type == 'lcg':
        if bits_to_keep is not None:
            expected_bits_to_keep = int(np.ceil(np.log2(m)))
            if bits_to_keep != expected_bits_to_keep:
                return False, f"LCG constraint: bits_to_keep ({bits_to_keep}) should equal bit length of m ({expected_bits_to_keep})"

    # Check bits_to_keep constraint for all types that use it
    if bits_to_keep is not None:
        bit_length = int(np.ceil(np.log2(m)))
        if not (bits_to_keep > 0 and bits_to_keep <= bit_length):
            return False, f"Invalid bits_to_keep: must be 0 < {bits_to_keep} <= {bit_length}"
    
    # Then check type-specific constraints for PCG variants
    if prng_type == 'pcg_rs':
        is_valid, error_msg = validate_pcg_rs_constraints(m, control_bits, bits_to_keep)
        if not is_valid:
            return False, error_msg
    
    elif prng_type == 'pcg_rr':
        is_valid, error_msg = validate_pcg_rr_constraints(m, control_bits, bits_to_keep)
        if not is_valid:
            return False, error_msg
    
    elif prng_type == 'pcg_xsh_rr':
        is_valid, error_msg = validate_pcg_xsh_rr_constraints(m, control_bits, bits_to_keep)
        if not is_valid:
            return False, error_msg
    
    elif prng_type == 'pcg_xsh_rs':
        is_valid, error_msg = validate_pcg_xsh_rs_constraints(control_bits, bits_to_keep)
        if not is_valid:
            return False, error_msg
    
    
    return True, ""





class ParameterBasedPRNGDataset(Dataset):
    """
    Dataset that generates PRNG sequences on-demand from stored parameters.
    No caching - generates sequences on-demand for maximum memory efficiency.
    """
    
    def __init__(self, param_sets: List[Dict]):
        """
        Args:
            param_sets: List of parameter dictionaries, each containing
                       the parameters needed to generate a sequence
        """
        self.param_sets = param_sets
        
        # Type mapping for PRNG functions
        self.type_mapping = {
            'lcg': base_b_lcg,
            'truncated_lcg': base_tlcg, 
            'tlcg': base_tlcg,
            'rs': base_b_pcg_rs,
            'rr': base_b_pcg_rr,
            'xshrr': base_b_pcg_xsh_rr,
            'xshrs': base_b_pcg_xsh_rs,
            'xslrr': base_b_pcg_xsl_rr,
        }
    
    def __len__(self) -> int:
        return len(self.param_sets)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate and return sequence for given index"""
        if idx >= len(self.param_sets) or idx < 0:
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.param_sets)}")
        
        # Generate on-demand
        param_set = self.param_sets[idx]
        sequence = self._generate_sequence_from_params(param_set, idx)
        
        # Ensure sequence is 1D
        if sequence.ndim > 1:
            sequence = sequence.flatten()
        
        # Convert to input/target tensors
        x = torch.tensor(sequence[:-1], dtype=torch.long)
        y = torch.tensor(sequence[1:], dtype=torch.long)
        
        return x, y
    
    def _generate_sequence_from_params(self, param_set: Dict, idx: int) -> np.ndarray:
        """Generate sequence from parameter dictionary"""
        prng_type = param_set['prng_type']
        prng_func = self.type_mapping.get(prng_type)
        
        if prng_func is None:
            raise ValueError(f"Unknown PRNG type: {prng_type}")
        
        # Create deterministic seed from index and parameters
        # This ensures same sequence for same index across calls
        seed_base = hash((idx, param_set['a'], param_set['c'], param_set['m'])) % (2**31)
        rng = np.random.default_rng(seed_base)
        
        # Generate x0
        x0 = rng.integers(1, param_set['m'])
        
        # Extract function parameters
        func_params = {k: v for k, v in param_set.items() 
                      if k not in ['prng_type', 'a', 'c']}
        
        # Generate sequence
        try:
            # The PRNG functions have different signatures, need to call them properly
            if prng_type == 'lcg':
                # base_b_lcg(m, seq_len, a, c, base, digits, num_examples, rng)
                sequence = prng_func(
                    m=param_set['m'],
                    seq_len=param_set['seq_len'],
                    a=param_set['a'],
                    c=param_set['c'],
                    base=param_set['base'],
                    digits=param_set['digits'],
                    num_examples=1,
                    rng=rng
                )
                # If num_examples=1, convert_to_base_b returns shape (1, seq_len), so take [0]
                if sequence.ndim > 1 and sequence.shape[0] == 1:
                    sequence = sequence[0]
            elif prng_type in ['truncated_lcg', 'tlcg']:
                # base_tlcg(m, seq_len, a, c, bits_to_keep, base, digits, num_examples, rng)
                sequence = prng_func(
                    m=param_set['m'],
                    seq_len=param_set['seq_len'],
                    a=param_set['a'],
                    c=param_set['c'],
                    bits_to_keep=param_set.get('bits_to_keep', 8),
                    base=param_set['base'],
                    digits=param_set['digits'],
                    num_examples=1,
                    rng=rng
                )
                if sequence.ndim > 1 and sequence.shape[0] == 1:
                    sequence = sequence[0]
            else:
                # PCG variants: func(m, seq_len, a, c, base, digits, control_bits, bits_to_keep, rng)
                # Note: PCG functions don't take x0 - they generate it internally
                sequence = prng_func(
                    m=param_set['m'],
                    seq_len=param_set['seq_len'],
                    a=param_set['a'],
                    c=param_set['c'],
                    base=param_set['base'],
                    digits=param_set['digits'],
                    control_bits=param_set.get('control_bits', 0),
                    bits_to_keep=param_set.get('bits_to_keep', 8),
                    num_examples=1,
                    rng=rng
                )
                # If num_examples=1, convert_to_base_b returns shape (1, seq_len), so take [0]
                if sequence.ndim > 1 and sequence.shape[0] == 1:
                    sequence = sequence[0]
            
            return sequence
            
        except Exception as e:
            raise RuntimeError(f"Error generating sequence for index {idx} with params {param_set}: {e}")


def generate_param_sets(config, rng, master_process=True, excluded_a=None, excluded_c=None) -> Tuple[List[Dict], Dict[str, List[Dict]], List[int], List[int], List[int], List[int]]:
    """
    Generate parameter sets with support for multi-type configurations and per-type evaluation.
    
    Args:
        config: Configuration object
        rng: Random number generator  
        master_process: Whether this is the master process (for logging)
        excluded_a: Set of a values to exclude from generation
        excluded_c: Set of c values to exclude from generation
        
    Returns:
        Tuple of (train_param_sets, test_param_sets_dict, train_a, train_c, val_a, val_c)
        where test_param_sets_dict maps type labels to individual test parameter sets
        For single-type configurations, test_param_sets_dict contains one entry
    """
    from .prng_data import find_as, find_coprimes
    
    t0 = time.time()
    
    # Parse control_bits
    if isinstance(config.control_bits, str) and ',' in config.control_bits:
        control_bits_list = [int(x.strip()) for x in config.control_bits.split(',')]
    else:
        control_bits_list = [int(config.control_bits)]
    
    # Handle multiple types
    if hasattr(config, 'type_list'):
        types_to_process = config.type_list
    else:
        types_to_process = [config.type]
    
    if master_process:
        print("="*80)
        print(f"GENERATING PARAM SETS: {'+'.join(types_to_process)} with m={config.m}")
        print(f"Control bits: {control_bits_list}")
        print("="*80)
    
    # Initialize excluded sets if not provided
    if excluded_a is None:
        excluded_a = set()
    if excluded_c is None:
        excluded_c = set()
    
    # Generate a and c values, accounting for excluded values
    buffer_factor = 3  # Generate extra to account for exclusions
    total_a_needed = config.n_a + config.n_test_a
    total_c_needed = config.n_c + config.n_test_c
    
    a_list = find_as(config.m, rng=rng, num=total_a_needed * buffer_factor)
    c_list = find_coprimes(config.m, rng=rng, num=total_c_needed * buffer_factor)
    
    # Filter out excluded values
    if excluded_a:
        a_list = [a for a in a_list if a not in excluded_a]
        if master_process:
            print(f"Excluded {len(excluded_a)} a values, {len(a_list)} remaining")
    
    if excluded_c:
        c_list = [c for c in c_list if c not in excluded_c]
        if master_process:
            print(f"Excluded {len(excluded_c)} c values, {len(c_list)} remaining")
    
    assert len(a_list) >= total_a_needed, f"not enough a values: needed {total_a_needed}, got {len(a_list)}"
    assert len(c_list) >= total_c_needed, f"not enough c values: needed {total_c_needed}, got {len(c_list)}"
    
    train_a, val_a = a_list[:config.n_a], a_list[config.n_a:config.n_a+config.n_test_a]
    train_c, val_c = c_list[:config.n_c], c_list[config.n_c:config.n_c+config.n_test_c]
    
    # Generate parameter sets for all types and control_bits combinations
    train_param_sets = []
    test_param_sets_dict = {}  # Maps type labels to their test parameter sets
    
    for current_type in types_to_process:
        if master_process:
            print(f"Processing type: {current_type}")
        
        # Common parameters
        common_params = {
            'prng_type': current_type.lower(),
            'm': config.m,
            'seq_len': config.seq_len,
            'base': config.base,
            'digits': config.digits
        }
        
        # Add bits_to_keep for types that use it
        if current_type not in ['LCG']:
            common_params['bits_to_keep'] = config.bits_to_keep
        
        # Handle control_bits for PCG variants
        if current_type in ['RS', 'RR', 'XSHRR', 'XSHRS', 'XSLRR']:
            for control_bits_val in control_bits_list:
                type_label = f"{current_type}_cb{control_bits_val}"
                params_with_cb = {**common_params, 'control_bits': control_bits_val}
                
                # Validate parameters for this PCG variant; skip if invalid
                validate_type_map = {
                    'RS': 'pcg_rs',
                    'RR': 'pcg_rr',
                    'XSHRR': 'pcg_xsh_rr',
                    'XSHRS': 'pcg_xsh_rs',
                    'XSLRR': 'pcg_xsl_rr',
                }
                v_type = validate_type_map[current_type]
                is_valid, err_msg = validate_prng_parameters(
                    v_type,
                    m=config.m,
                    control_bits=control_bits_val,
                    bits_to_keep=params_with_cb.get('bits_to_keep')
                )
                if not is_valid:
                    if master_process:
                        print(f"Skipping {type_label} due to invalid parameters: {err_msg}")
                    continue
                
                # Generate parameter sets for all (a,c) combinations for training
                for a in train_a:
                    for c in train_c:
                        for _ in range(config.n_example):
                            param_set = {**params_with_cb, 'a': a, 'c': c}
                            train_param_sets.append(param_set)
                
                # Generate test parameter sets for this specific type/control_bits combo
                type_test_param_sets = []
                for a in val_a:
                    for c in val_c:
                        param_set = {**params_with_cb, 'a': a, 'c': c}
                        type_test_param_sets.append(param_set)
                test_param_sets_dict[type_label] = type_test_param_sets
        
        elif current_type == 'XSPCGs':
            # Handle XSPCGs (xorshift PCG variants)
            for control_bits_val in control_bits_list:
                # Define xorshift PCG types  
                pcg_types = ['xslrr', 'xshrr']
                
                # Check if PCG_XSH_RS constraint is satisfied before including it
                if validate_pcg_xsh_rs_constraints(control_bits_val, config.bits_to_keep):
                    pcg_types.append('xshrs')
                
                for pcg_subtype in pcg_types:
                    type_label = f"{pcg_subtype.upper()}_cb{control_bits_val}"
                    params_with_cb = {**common_params, 'control_bits': control_bits_val, 'prng_type': pcg_subtype}
                    
                    # Validate parameters for this XSPCGs subtype; skip if invalid
                    validate_type_map = {
                        'xslrr': 'pcg_xsl_rr',
                        'xshrr': 'pcg_xsh_rr',
                        'xshrs': 'pcg_xsh_rs',
                    }
                    v_type = validate_type_map[pcg_subtype]
                    is_valid, err_msg = validate_prng_parameters(
                        v_type,
                        m=config.m,
                        control_bits=control_bits_val,
                        bits_to_keep=params_with_cb.get('bits_to_keep')
                    )
                    if not is_valid:
                        if master_process:
                            print(f"Skipping {type_label} due to invalid parameters: {err_msg}")
                        continue
                    
                    # Generate parameter sets for training
                    for a in train_a:
                        for c in train_c:
                            for _ in range(config.n_example):
                                param_set = {**params_with_cb, 'a': a, 'c': c}
                                train_param_sets.append(param_set)
                    
                    # Generate test parameter sets for this specific type/control_bits combo
                    type_test_param_sets = []
                    for a in val_a:
                        for c in val_c:
                            param_set = {**params_with_cb, 'a': a, 'c': c}
                            type_test_param_sets.append(param_set)
                    test_param_sets_dict[type_label] = type_test_param_sets
        
        else:
            # Single types like LCG, TLCG
            type_label = current_type
            
            # Generate parameter sets for training
            for a in train_a:
                for c in train_c:
                    for _ in range(config.n_example):
                        param_set = {**common_params, 'a': a, 'c': c}
                        train_param_sets.append(param_set)
            
            # Generate test parameter sets for this type
            type_test_param_sets = []
            for a in val_a:
                for c in val_c:
                    param_set = {**common_params, 'a': a, 'c': c}
                    type_test_param_sets.append(param_set)
            test_param_sets_dict[type_label] = type_test_param_sets
    
    if master_process:
        print(f"Generated {len(train_param_sets)} training parameter sets")
        print(f"Generated test parameter sets:")
        for type_label, type_sets in test_param_sets_dict.items():
            print(f"  - {type_label}: {len(type_sets)} sets")
        print(f"Time taken: {time.time()-t0:.2f} seconds")
    
    return train_param_sets, test_param_sets_dict, train_a, train_c, val_a, val_c


def generate_lowmem_data(config, rng, master_process=True, excluded_a=None, excluded_c=None) -> Tuple[Dataset, Dataset, List[int], List[int], List[int], List[int], Dict[str, Dataset]]:
    """
    Memory-efficient (lowmem) replacement for generate_data() that uses on-demand sequence generation.
    
    Args:
        config: Configuration object
        rng: Random number generator
        master_process: Whether this is the master process
        excluded_a: Set of a values to exclude from generation
        excluded_c: Set of c values to exclude from generation
        
    Returns:
        Tuple of (train_dataset, test_dataset, train_a, train_c, val_a, val_c, per_type_test_datasets)
        where per_type_test_datasets is a dict mapping type labels to individual test datasets
    """
    t0 = time.time()
    
    # Check if we have multiple types that need per-type evaluation
    has_multiple_types = False
    if hasattr(config, 'type_list') and len(config.type_list) > 1:
        has_multiple_types = True
    elif '+' in config.type:
        has_multiple_types = True
    elif isinstance(config.control_bits, str) and ',' in config.control_bits:
        has_multiple_types = True
    elif any(t in config.type for t in ['PCGs', 'XSPCGs']):
        has_multiple_types = True
    
    if has_multiple_types:
        # Generate parameter sets organized by type for detailed evaluation
        train_param_sets, test_param_sets_dict, train_a, train_c, val_a, val_c = generate_param_sets(config, rng, master_process, excluded_a, excluded_c)
        
        # Create combined train dataset
        train_dataset = ParameterBasedPRNGDataset(train_param_sets)
        
        # Create combined test dataset and individual per-type test datasets
        all_test_param_sets = []
        per_type_test_datasets = {}
        
        for type_label, type_param_sets in test_param_sets_dict.items():
            # Create individual per-type test dataset
            per_type_test_datasets[type_label] = ParameterBasedPRNGDataset(type_param_sets)
            # Add to combined test dataset
            all_test_param_sets.extend(type_param_sets)
        
        test_dataset = ParameterBasedPRNGDataset(all_test_param_sets)
        
        if master_process:
            print("-"*80)
            print(f"LOWMEM DATA GENERATION COMPLETE (with per-type):")
            print(f"  - Train dataset size: {len(train_dataset)} sequences")
            print(f"  - Combined test dataset size: {len(test_dataset)} sequences")
            print(f"  - Per-type test datasets: {list(per_type_test_datasets.keys())}")
            for label, dataset in per_type_test_datasets.items():
                print(f"    - {label}: {len(dataset)} sequences")
            print(f"  - Time taken: {time.time()-t0:.2f} seconds")
            print(f"  - Memory usage: Parameters only, no caching")
            print("-"*80)
    else:
        # Single type - use main method
        train_param_sets, test_param_sets_dict, train_a, train_c, val_a, val_c = generate_param_sets(config, rng, master_process, excluded_a, excluded_c)
        
        # Create lowmem datasets with no caching
        train_dataset = ParameterBasedPRNGDataset(train_param_sets)
        
        # For single type, test_param_sets_dict contains one entry
        test_param_sets = list(test_param_sets_dict.values())[0]  # Get the single test set
        test_dataset = ParameterBasedPRNGDataset(test_param_sets)
        per_type_test_datasets = {}  # Empty for single type
        
        if master_process:
            print("-"*80)
            print(f"LOWMEM DATA GENERATION COMPLETE (single type):")
            print(f"  - Train dataset size: {len(train_dataset)} sequences")
            print(f"  - Test dataset size: {len(test_dataset)} sequences")
            print(f"  - Time taken: {time.time()-t0:.2f} seconds")
            print(f"  - Memory usage: Parameters only, no caching")
            print("-"*80)
    return train_dataset, test_dataset, train_a, train_c, val_a, val_c, per_type_test_datasets


def create_multi_modulus_lowmem_datasets(moduli_configs: List, base_rng_seed: int = 97) -> Tuple[List[Dataset], List[Dataset]]:
    """
    Create memory-efficient (lowmem) datasets for multiple moduli configurations.
    
    Args:
        moduli_configs: List of configuration objects, one per modulus
        base_rng_seed: Base seed for RNG

        
    Returns:
        Tuple of (train_datasets, test_datasets) lists
    """
    train_datasets = []
    test_datasets = []
    
    for i, config in enumerate(moduli_configs):
        rng = np.random.default_rng(base_rng_seed + i * 1000)
        train_dataset, test_dataset, _, _, _, _, _ = generate_lowmem_data(
            config, rng, master_process=False
        )
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)
    
    return train_datasets, test_datasets


def create_curriculum_lowmem_datasets(config, master_process: bool = True, 
                                        ddp: bool = False, 
                                        rank: int = None, world_size: int = None,
                                        num_workers: int = 4,
                                        excluded_a: set = None, excluded_c: set = None) -> Tuple[List[Dataset], List[DataLoader], List[List[int]], List[List[int]]]:
    """
    Create memory-efficient (lowmem) datasets for curriculum learning.
    
    Args:
        config: Configuration object with moduli list and other parameters
        master_process: Whether this is the master process
        ddp: Whether using distributed data parallel
        rank: Process rank for distributed training
        world_size: Total number of processes for distributed training
        num_workers: Number of DataLoader workers
        excluded_a: Set of a values to exclude from generation
        excluded_c: Set of c values to exclude from generation
        
    Returns:
        Tuple of (train_datasets, test_loaders, train_a_values, train_c_values) where:
        - train_datasets are lowmem datasets
        - test_loaders are DataLoader instances
        - train_a_values is a list of training a values for each modulus
        - train_c_values is a list of training c values for each modulus
    """
    train_datasets = []
    test_loaders = []
    train_a_values = []
    train_c_values = []
    
    for i, m in enumerate(config.moduli):
        if master_process:
            print(f"Creating dataset for modulus {m} ({i+1}/{len(config.moduli)})")
        
        # Create config for this modulus (same logic as original)
        config_m = argparse.Namespace(**vars(config))
        config_m.m = m
        
        # Set bits_to_keep
        if hasattr(config, 'moduli_bits_to_keep') and config.moduli_bits_to_keep is not None:
            if i < len(config.moduli_bits_to_keep):
                config_m.bits_to_keep = config.moduli_bits_to_keep[i]
            else:
                config_m.bits_to_keep = int(np.ceil(np.log2(m)))
        else:
            config_m.bits_to_keep = int(np.ceil(np.log2(m)))
        
        # Generate dataset with different seed for each modulus
        rng = np.random.default_rng(config.data_seed + i * 1000)
        train_dataset, test_dataset, train_a, train_c, _, _, _ = generate_lowmem_data(
            config_m, rng, master_process=False, excluded_a=excluded_a, excluded_c=excluded_c
        )
        
        train_datasets.append(train_dataset)
        train_a_values.append(train_a)
        train_c_values.append(train_c)
        
        # Create test loader
        if ddp:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=config.batch_size,
                sampler=test_sampler,
                pin_memory=True,
                drop_last=True,
                num_workers=num_workers,
                prefetch_factor=2
            )
        else:
            test_loader = DataLoader(
                test_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                pin_memory=True,
                drop_last=True,
                num_workers=num_workers,
                prefetch_factor=2
            )
        
        test_loaders.append(test_loader)
    
    if master_process:
        total_train = sum(len(ds) for ds in train_datasets)
        total_test = sum(len(loader.dataset) for loader in test_loaders)
        print("Curriculum datasets created:")
        print(f"  - {len(config.moduli)} moduli")
        print(f"  - Total train sequences: {total_train}")
        print(f"  - Total test sequences: {total_test}")
        print(f"  - Memory saved: ~{total_train * config.seq_len * 8 / 1024**2:.1f} MB")
        print(f"  - No caching: Pure on-demand generation")
    
    return train_datasets, test_loaders, train_a_values, train_c_values
