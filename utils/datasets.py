import torch
from torch.utils.data import Dataset, ConcatDataset
import torch.nn.functional as F
import numpy as np
from utils.prng_data import *
from typing import List, Optional, Callable, Union


def validate_bits_to_keep(m: int, bits_to_keep: int) -> bool:
    """Validate bits_to_keep parameter for truncated generators"""
    if bits_to_keep is None:
        return False
    bit_length = int(np.ceil(np.log2(m)))
    return bits_to_keep > 0 and bits_to_keep <= bit_length


def validate_pcg_rs_constraints(m: int, control_bits: int, bits_to_keep: int) -> bool:
    """Validate PCG_RS constraints: bit_length > (2 ** control_bits - 1) + bits_to_keep"""
    if control_bits is None or bits_to_keep is None:
        return False
    bit_length = int(np.ceil(np.log2(m)))
    return bit_length > (2 ** control_bits - 1) + bits_to_keep


def validate_pcg_rr_constraints(m: int, control_bits: int, bits_to_keep: int) -> bool:
    """Validate PCG_RR constraints: bits_to_keep >= 2 ** control_bits"""
    if control_bits is None or bits_to_keep is None:
        return False
    return bits_to_keep >= 2 ** control_bits


def validate_pcg_xsh_rr_constraints(m: int, control_bits: int, bits_to_keep: int) -> bool:
    """Validate PCG_XSH_RR constraints: bit_length > control_bits + bits_to_keep"""
    if control_bits is None or bits_to_keep is None:
        return False
    bit_length = int(np.ceil(np.log2(m)))
    return bit_length > control_bits + bits_to_keep


def validate_pcg_xsh_rs_constraints(control_bits: int, bits_to_keep: int) -> bool:
    """Validate PCG_XSH_RS constraints: constant_shift > 0"""
    if control_bits is None or bits_to_keep is None:
        return False
    constant_shift = bits_to_keep - control_bits - 2 ** control_bits + 1
    return constant_shift > 0




def validate_prng_parameters(prng_type: str, **kwargs) -> tuple[bool, str]:
    """
    Validate parameters for a specific PRNG type.
    Returns (is_valid, error_message)
    """
    m = kwargs.get('m')
    control_bits = kwargs.get('control_bits')
    bits_to_keep = kwargs.get('bits_to_keep')
    
    # Special validation for LCG: bits_to_keep should equal the bit length of m
    if prng_type == 'lcg':
        if m is not None and bits_to_keep is not None:
            expected_bits_to_keep = int(np.ceil(np.log2(m)))
            if bits_to_keep != expected_bits_to_keep:
                return False, f"LCG constraint: bits_to_keep ({bits_to_keep}) should equal bit length of m ({expected_bits_to_keep})"
        if control_bits is not None and control_bits != 0:
            return False, f"LCG constraint: control_bits ({control_bits}) should be 0"
    # TLCG does not use control_bits
    if prng_type == 'truncated_lcg':
        if control_bits is not None and control_bits != 0:
            return False, f"TLCG constraint: control_bits ({control_bits}) should be 0"
    
    # Check bits_to_keep constraint for all types that use it
    if bits_to_keep is not None:
        bit_length = int(np.ceil(np.log2(m)))
        if not (bits_to_keep > 0 and bits_to_keep <= bit_length):
            return False, f"Invalid bits_to_keep: must be 0 < {bits_to_keep} <= {bit_length}"
    
    # Then check type-specific constraints for PCG variants
    if prng_type == 'pcg_rs':
        if m is not None and control_bits is not None and bits_to_keep is not None:
            if not validate_pcg_rs_constraints(m, control_bits, bits_to_keep):
                bit_length = int(np.ceil(np.log2(m)))
                required = (2 ** control_bits - 1) + bits_to_keep
                return False, f"PCG_RS constraint violated: bit_length ({bit_length}) must be > {required}"
    
    elif prng_type == 'pcg_rr':
        if control_bits is not None and bits_to_keep is not None:
            if not validate_pcg_rr_constraints(m, control_bits, bits_to_keep):
                required = 2 ** control_bits
                return False, f"PCG_RR constraint violated: bits_to_keep ({bits_to_keep}) must be >= {required}"
    
    elif prng_type == 'pcg_xsh_rr':
        if m is not None and control_bits is not None and bits_to_keep is not None:
            if not validate_pcg_xsh_rr_constraints(m, control_bits, bits_to_keep):
                bit_length = int(np.ceil(np.log2(m)))
                required = control_bits + bits_to_keep
                return False, f"PCG_XSH_RR constraint violated: bit_length ({bit_length}) must be > {required}"
    
    elif prng_type == 'pcg_xsh_rs':
        if control_bits is not None and bits_to_keep is not None:
            if not validate_pcg_xsh_rs_constraints(control_bits, bits_to_keep):
                constant_shift = bits_to_keep - control_bits - 2 ** control_bits + 1
                return False, f"PCG_XSH_RS constraint violated: constant_shift ({constant_shift}) must be > 0"
    
    
    return True, ""


class BasePRNGDataset(Dataset):
    """
    Base class for PRNG datasets that handles the common pattern of:
    1. Creating sequences from meshgrid of (a, c) parameters
    2. Concatenating sequences 
    3. Providing PyTorch Dataset interface
    """

    def __init__(
            self,
        prng_func: Callable,
        a_list: List[int],
        c_list: List[int],
        num_examples: Optional[int] = None,
        rng: np.random.Generator = np.random.default_rng(97),
        **kwargs
    ):
        """
        Args:
            prng_func: The PRNG function to use for sequence generation
            a_list: List of 'a' parameters for the PRNG
            c_list: List of 'c' parameters for the PRNG  
            num_examples: Number of examples per (a,c) pair
            rng: Random number generator
            **kwargs: Additional parameters passed to prng_func
        """
        self.prng_func = prng_func
        self.kwargs = kwargs
        
        # Create all (a,c) combinations
        a_mesh, c_mesh = np.meshgrid(a_list, c_list)
        a_flat = a_mesh.flatten()
        c_flat = c_mesh.flatten()
        ac_pairs = np.vstack((a_flat, c_flat)).T
        
        # Generate sequences for each (a,c) pair
        # Use NumPy throughout and convert once to torch to avoid back-and-forth copies
        sequences_np = []
        for a, c in ac_pairs:
            seq = prng_func(a=a, c=c, num_examples=num_examples, rng=rng, **kwargs)
            if isinstance(seq, torch.Tensor):
                seq = seq.detach().cpu().numpy()
            else:
                seq = np.asarray(seq, dtype=np.int64)
            sequences_np.append(seq)
        self.sequences = torch.from_numpy(np.concatenate(sequences_np, axis=0))
    
    def __len__(self) -> int:
        return self.sequences.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (input, target) where target is input shifted by 1 position"""
        x, y = self.sequences[idx][:-1], self.sequences[idx][1:]
        return x, y


# Registry of PRNG functions - Always use base-b variants
PRNG_REGISTRY = {
    'lcg': base_b_lcg,  # Base-b version
    'truncated_lcg': base_tlcg,  # Base-b version  
    'pcg_rs': base_b_pcg_rs,  # Base-b version
    'pcg_rr': base_b_pcg_rr,  # Base-b version
    'pcg_xsh_rr': base_b_pcg_xsh_rr,  # Base-b version
    'pcg_xsh_rs': base_b_pcg_xsh_rs,  # Base-b version
    'pcg_xsl_rr': base_b_pcg_xsl_rr,  # Base-b version
}


def create_prng_dataset(
    prng_type: str,
    a_list: List[int],
    c_list: List[int],
    num_examples: Optional[int] = None,
    rng: np.random.Generator = np.random.default_rng(97),
    validate_params: bool = True,
    **kwargs
) -> BasePRNGDataset:
    """
    Factory function to create PRNG datasets.
    Always uses base-b representation with automatic defaults:
    - For LCG: base = m (if not specified)
    - For PCG/TLCG: base = 2**bits_to_keep (if not specified)  
    - digits = 1 (if not specified) makes it equivalent to regular representation
    
    Args:
        prng_type: Type of PRNG ('lcg', 'truncated_lcg', 'pcg_rs', etc.)
        a_list: List of 'a' parameters
        c_list: List of 'c' parameters
        num_examples: Number of examples per (a,c) pair
        rng: Random number generator
        validate_params: Whether to validate parameters before creation
        **kwargs: Additional parameters for the specific PRNG function
    
    Returns:
        BasePRNGDataset instance
    """
    if prng_type not in PRNG_REGISTRY:
        raise ValueError(f"Unknown PRNG type: {prng_type}. Available: {list(PRNG_REGISTRY.keys())}")
    
    # Set default base and digits if not provided
    if 'base' not in kwargs or kwargs['base'] is None:
        if prng_type == 'lcg':
            kwargs['base'] = kwargs.get('m')
        else:  # PCG and TLCG types
            kwargs['base'] = 2 ** kwargs.get('bits_to_keep')
    
    if 'digits' not in kwargs or kwargs['digits'] is None:
        kwargs['digits'] = 1
    
    # Validate parameters if requested
    if validate_params:
        is_valid, error_msg = validate_prng_parameters(prng_type, **kwargs)
        if not is_valid:
            raise ValueError(f"Invalid parameters for {prng_type}: {error_msg}")
    
    # Get the PRNG function and create dataset
    prng_func = PRNG_REGISTRY[prng_type]
    return BasePRNGDataset(prng_func, a_list, c_list, num_examples, rng, **kwargs)




def generate_data(config, rng, master_process):
    """Generate training and testing datasets based on configuration"""
    import time
    from torch.utils.data import ConcatDataset
    from .prng_data import find_as, find_coprimes
    
    t0 = time.time()
    
    # Parse control_bits - support comma-separated values like "2,3"
    if isinstance(config.control_bits, str) and ',' in config.control_bits:
        control_bits_list = [int(x.strip()) for x in config.control_bits.split(',')]
    else:
        control_bits_list = [int(config.control_bits)]
    
    # Handle multiple types separated by '+' 
    if hasattr(config, 'type_list'):
        types_to_process = config.type_list
        if master_process:
            print("="*80)
            print(f"GENERATING DATA: Multiple types {'+'.join(types_to_process)} with m={config.m}, bits_to_keep={config.bits_to_keep}")
            print(f"Control bits: {control_bits_list}")
            print(f"Generating {config.n_a} training 'a' values and {config.n_c} training 'c' values")
            print(f"Generating {config.n_test_a} test 'a' values and {config.n_test_c} test 'c' values")
            print("="*80)
    else:
        types_to_process = [config.type]
        if master_process:
            print("="*80)
            print(f"GENERATING DATA: {config.type} with m={config.m}, bits_to_keep={config.bits_to_keep}")
            print(f"Control bits: {control_bits_list}")
            print(f"Generating {config.n_a} training 'a' values and {config.n_c} training 'c' values")
            print(f"Generating {config.n_test_a} test 'a' values and {config.n_test_c} test 'c' values")
            print("="*80)
    
    # Generate a and c values using proper mathematical functions
    a_list = find_as(config.m, rng=rng, num=config.n_a+config.n_test_a)
    c_list = find_coprimes(config.m, rng=rng, num=config.n_c+config.n_test_c)
    assert len(a_list) >= config.n_a+config.n_test_a, "not enough a values"
    assert len(c_list) >= config.n_c+config.n_test_c, "not enough c values"
    train_a, val_a = a_list[:config.n_a], a_list[config.n_a:]
    train_c, val_c = c_list[:config.n_c], c_list[config.n_c:]
    
    # Initialize lists to collect datasets from all types
    all_train_datasets = []
    all_test_datasets = []
    
    # Process each type
    for current_type in types_to_process:
        if master_process:
            print(f"Processing type: {current_type}")
        
        # Create a temporary config-like object for this specific type
        # We'll use the original config but override the type
        import copy
        temp_config = copy.deepcopy(config)
        temp_config.type = current_type
        
        # Generate datasets for this specific type using the existing logic
        train_ds, test_ds = generate_single_type_data(temp_config, rng, master_process, 
                                                      train_a, train_c, val_a, val_c, 
                                                      control_bits_list)
        all_train_datasets.append(train_ds)
        all_test_datasets.append(test_ds)
    
    # Concatenate all datasets
    if len(all_train_datasets) > 1:
        train_dataset = ConcatDataset(all_train_datasets)
        test_dataset = ConcatDataset(all_test_datasets)
    else:
        train_dataset = all_train_datasets[0]
        test_dataset = all_test_datasets[0]
    
    t1 = time.time()
    if master_process:
        print("-"*80)
        print(f"DATA GENERATION COMPLETE:")
        print(f"  - Train dataset size: {len(train_dataset)} sequences")
        print(f"  - Test dataset size: {len(test_dataset)} sequences")
        print(f"  - Time taken: {t1-t0:.2f} seconds")
        print("-"*80)
    
    return train_dataset, test_dataset, train_a, train_c, val_a, val_c


def generate_single_type_data(config, rng, master_process, train_a, train_c, val_a, val_c, control_bits_list):
    """Generate datasets for a single PRNG type"""
    from torch.utils.data import ConcatDataset
    
    # Map config types to PRNG registry names (all use base-b representation now)
    type_mapping = {
        'RS': 'pcg_rs',
        'RR': 'pcg_rr', 
        'XSHRR': 'pcg_xsh_rr',
        'XSHRS': 'pcg_xsh_rs',
        'XSLRR': 'pcg_xsl_rr',
        'LCG': 'lcg',
        'TLCG': 'truncated_lcg'
    }
    
    # Generate datasets based on config type (unified base-b approach)
    if config.type in type_mapping:
        prng_type = type_mapping[config.type]
        
        # Common parameters for all datasets
        common_params = {
            'm': config.m,
            'seq_len': config.seq_len,
            'base': config.base,
            'digits': config.digits
        }
        
        # Add bits_to_keep for types that use it (all except fixed-output PCG variants)
        if config.type not in ['RXSMXS1616', 'LCG']:
            common_params['bits_to_keep'] = config.bits_to_keep
            
        # Store original config values for informational messages
        original_control_bits = control_bits_list[0] if control_bits_list else None
        original_bits_to_keep = config.bits_to_keep
        

        
        # Handle control_bits for PCG variants
        if config.type in ['RS', 'RR', 'XSHRR', 'XSHRS', 'XSLRR'] and len(control_bits_list) == 1:
            # Single control_bits value
            common_params['control_bits'] = control_bits_list[0]
            train_dataset = create_prng_dataset(
                prng_type, train_a, train_c, num_examples=config.n_example, rng=rng, **common_params
            )
            test_dataset = create_prng_dataset(
                prng_type, val_a, val_c, num_examples=1, rng=rng, **common_params
            )
        elif config.type in ['RS', 'RR', 'XSHRR', 'XSHRS', 'XSLRR'] and len(control_bits_list) > 1:
            # Multiple control_bits values - create multiple datasets and concatenate
            train_dataset = list()
            test_dataset = list()
            
            for control_bits_val in control_bits_list:
                params_with_cb = {**common_params, 'control_bits': control_bits_val}
                train_dataset.append(create_prng_dataset(
                    prng_type, train_a, train_c, num_examples=config.n_example, rng=rng, **params_with_cb
                ))
                test_dataset.append(create_prng_dataset(
                    prng_type, val_a, val_c, num_examples=1, rng=rng, **params_with_cb
                ))
            
            train_dataset = ConcatDataset(train_dataset)
            test_dataset = ConcatDataset(test_dataset)
        else:
            # Simple case: LCG, TLCG, and fixed-output PCG variants
            if config.type == 'LCG':
                # Create filtered parameters for LCG
                lcg_params = {k: v for k, v in common_params.items() 
                             if k not in ['bits_to_keep', 'control_bits']}
                
                # Print informational message if multi-type config has these parameters
                if hasattr(config, 'type_list') and len(config.type_list) > 1:
                    expected_bits_to_keep = int(np.ceil(np.log2(config.m)))
                    
                    if original_bits_to_keep is not None or original_control_bits is not None:
                        if master_process:
                            print(f"Info: For LCG in multi-type configuration:")
                            if original_control_bits is not None:
                                print(f"  - LCG uses control_bits = 0 (config has {original_control_bits})")
                            if original_bits_to_keep is not None:
                                print(f"  - LCG uses effective bits_to_keep = {expected_bits_to_keep} (bit length of m, config has {original_bits_to_keep})")
                
                train_dataset = create_prng_dataset(
                    prng_type, train_a, train_c, num_examples=config.n_example, rng=rng, **lcg_params
                )
                test_dataset = create_prng_dataset(
                    prng_type, val_a, val_c, num_examples=1, rng=rng, **lcg_params
                )
            else:
                train_dataset = create_prng_dataset(
                    prng_type, train_a, train_c, num_examples=config.n_example, rng=rng, **common_params
                )
                test_dataset = create_prng_dataset(
                    prng_type, val_a, val_c, num_examples=1, rng=rng, **common_params
                )
    elif config.type == 'PCGs':
        # For "PCGs" case, include all PCG variants but exclude LCG and TLCG
        train_dataset = list()
        test_dataset = list()
        
        for control_bits_val in control_bits_list:
            # Define PCG types that require control_bits
            pcg_types = ['pcg_rs', 'pcg_rr', 'pcg_xsh_rr', 'pcg_xsl_rr']
            
            # Check if PCG_XSH_RS constraint is satisfied before including it
            if validate_pcg_xsh_rs_constraints(control_bits_val, config.bits_to_keep):
                pcg_types.append('pcg_xsh_rs')
            else:
                if master_process:
                    constant_shift = config.bits_to_keep - control_bits_val - 2 ** control_bits_val + 1
                    print(f"Warning: Skipping PCG_XSH_RS for control_bits={control_bits_val} - constraint not satisfied (bits_to_keep={config.bits_to_keep}, control_bits={control_bits_val}, constant_shift={constant_shift})")
            
            
            # Add PCG datasets for this control_bits value
            pcg_params = {
                'm': config.m,
                'seq_len': config.seq_len,
                'base': config.base,
                'digits': config.digits,
                'control_bits': control_bits_val,
                'bits_to_keep': config.bits_to_keep
            }
            for prng_type in pcg_types:
                train_dataset.append(create_prng_dataset(
                    prng_type, train_a, train_c, num_examples=config.n_example, rng=rng, **pcg_params
                ))
                test_dataset.append(create_prng_dataset(
                    prng_type, val_a, val_c, num_examples=1, rng=rng, **pcg_params
                ))
        
        train_dataset = ConcatDataset(train_dataset)
        test_dataset = ConcatDataset(test_dataset)

    elif config.type == 'XSPCGs':
        # For "XSPCGs" case, include only xorshift PCG variants
        train_dataset = list()
        test_dataset = list()
        
        for control_bits_val in control_bits_list:
            # Define xorshift PCG types
            pcg_types = ['pcg_xsl_rr', 'pcg_xsh_rr']
            
            # Check if PCG_XSH_RS constraint is satisfied before including it
            if validate_pcg_xsh_rs_constraints(control_bits_val, config.bits_to_keep):
                pcg_types.append('pcg_xsh_rs')
            else:
                if master_process:
                    constant_shift = config.bits_to_keep - control_bits_val - 2 ** control_bits_val + 1
                    print(f"Warning: Skipping PCG_XSH_RS for control_bits={control_bits_val} - constraint not satisfied (bits_to_keep={config.bits_to_keep}, control_bits={control_bits_val}, constant_shift={constant_shift})")
            
            # Add xorshift PCG datasets for this control_bits value
            pcg_params = {
                'm': config.m,
                'seq_len': config.seq_len,
                'base': config.base,
                'digits': config.digits,
                'control_bits': control_bits_val,
                'bits_to_keep': config.bits_to_keep
            }
            for prng_type in pcg_types:
                train_dataset.append(create_prng_dataset(
                    prng_type, train_a, train_c, num_examples=config.n_example, rng=rng, **pcg_params
                ))
                test_dataset.append(create_prng_dataset(
                    prng_type, val_a, val_c, num_examples=1, rng=rng, **pcg_params
                ))
        
        train_dataset = ConcatDataset(train_dataset)
        test_dataset = ConcatDataset(test_dataset)

    return train_dataset, test_dataset
