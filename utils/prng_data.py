import os
import pandas as pd
from sympy import primefactors, gcd
from itertools import *
import random
import numpy as np
from typing import Optional


def find_as(m: int, limit: Optional[int] = None, rng: np.random.Generator = np.random.default_rng(97), num: int = 2000) -> list:
    """Find 'a' values for LCG that satisfy the full period requirement"""
    if limit is None:
        limit = m
    factors = primefactors(m)

    if m % 4 == 0:
        factors.append(4)
        factors.remove(2)
    lcm = 1
    for factor in set(factors):
        lcm *= factor
    result = []
    max_possible = limit // lcm
    if max_possible <= 1:
        return []
    

    # Generate unique random multipliers (memory efficient)
    k_set = set()
    max_k = limit // lcm
    # Avoid infinite loop: only (max_k-1) unique k values are possible in [1, max_k)
    if num > max_k - 1:
        print(f"Warning: requested {num} 'a' values but only {max_k - 1} possible for m={m} (limit={limit}). Reducing to {max_k - 1}.")
        num = max(0, max_k - 1)
    
    while len(k_set) < num:
        k = rng.integers(1, max_k)
        k_set.add(k)

    for k in k_set:
        a = k * lcm + 1
        result.append(a)
    return result


def find_coprimes(m: int, rng: np.random.Generator = np.random.default_rng(97), num: int = 2000, low: int = 3) -> list:
    """Find numbers coprime to m, distributed across the full range [low, m-1]"""
    
    def search_power_of_2(retry_attempt: int = 0):
        """Search coprimes for m = 2^k (odd numbers only)"""
        coprimes = []
        start = low if low % 2 == 1 else low + 1
        
        # Increase attempts with each retry
        base_attempts = num * 10
        max_attempts = base_attempts * (2 ** retry_attempt)
        attempts = 0
        seen = set()
        
        while len(coprimes) < num and attempts < max_attempts:
            # Generate random even number, then add 1 to make it odd
            candidate = rng.integers(start // 2, m // 2) * 2 + 1
            if candidate >= start and candidate < m and candidate not in seen:
                coprimes.append(candidate)
                seen.add(candidate)
            attempts += 1
        
        return coprimes
    
    def search_general(retry_attempt: int = 0):
        """Search coprimes for general m using GCD"""
        coprimes = []
        base_attempts = num * 50
        max_attempts = base_attempts * (2 ** retry_attempt)
        attempts = 0
        
        while len(coprimes) < num and attempts < max_attempts:
            candidate = rng.integers(low, m)
            if gcd(candidate, m) == 1:
                coprimes.append(candidate)
            attempts += 1
        
        # Remove duplicates while preserving order
        seen = set()
        coprimes = [x for x in coprimes if not (x in seen or seen.add(x))]
        return coprimes
    
    # Check if m is a power of 2 for optimization
    is_power_of_2 = m > 0 and (m & (m - 1)) == 0
    
    if is_power_of_2:
        # Try power-of-2 optimization with retries
        for retry in range(3):  # Try up to 3 times
            coprimes = search_power_of_2(retry)
            if len(coprimes) >= num:
                break
            if retry < 2:  # Don't print warning on final attempt
                print(f"Retry {retry + 1}: Only found {len(coprimes)} coprimes, trying again with more attempts...")
                
    elif m <= 50000:
        # For small m, exhaustive search should always work
        all_numbers = np.arange(low, m)
        rng.shuffle(all_numbers)
        
        coprimes = []
        for i in all_numbers:
            if gcd(i, m) == 1:
                coprimes.append(i)
                if len(coprimes) >= num:
                    break
                    
        # If exhaustive search fails, there truly aren't enough coprimes
        if len(coprimes) < num:
            total_possible = sum(1 for i in range(low, m) if gcd(i, m) == 1)
            print(f"Warning: Only {total_possible} coprimes exist in range [{low}, {m-1}], but {num} requested")
            
    else:
        # Try general search with retries
        for retry in range(3):  # Try up to 3 times
            coprimes = search_general(retry)
            if len(coprimes) >= num:
                break
            if retry < 2:  # Don't print warning on final attempt
                print(f"Retry {retry + 1}: Only found {len(coprimes)} coprimes, trying again with more attempts...")
    
    if len(coprimes) < num:
        print(f"Warning: only found {len(coprimes)} coprimes out of {num} requested for m={m} after all retries")
    
    return coprimes[:num]  # Return exactly 'num' coprimes (or fewer if not enough found)





def lcg(m: int = 512, seq_len: int = 8, a: int = 45, c: int = 123, rng: np.random.Generator = np.random.default_rng(97), num_examples: Optional[int] = None) -> np.ndarray:
    """Linear Congruential Generator"""
    if num_examples:
        n = min(m // seq_len, num_examples)
    else:
        n = m // seq_len

    array = np.zeros(n * seq_len, dtype=np.int64)
    # Maintain state as Python int to avoid NumPy overflow warnings
    x = int(rng.integers(low=0, high=m))
    for i in range(n * seq_len):
        x = (int(a) * x + int(c)) % int(m)
        array[i] = x

    reshaped_array = array.reshape(n, seq_len)
    return reshaped_array


def base_b_lcg(m: int = 512, seq_len: int = 8, a: int = 45, c: int = 123, 
               base: int = 2, digits: int = 8, num_examples: int = 1, 
               rng: np.random.Generator = np.random.default_rng(97)) -> np.ndarray:
    """Linear Congruential Generator with base-b representation"""
    return convert_to_base_b(
        lcg, base=base, digits=digits,
        m=m, seq_len=seq_len, a=a, c=c, num_examples=num_examples, rng=rng
    )


def truncated_lcg(m: int = 512, seq_len: int = 8, a: int = 45, c: int = 123, bits_to_keep: int = 8, num_examples: int = 1, rng: np.random.Generator = np.random.default_rng(97)) -> np.ndarray:
    """Truncated Linear Congruential Generator"""
    array = np.zeros(num_examples * seq_len, dtype=np.int64)
    x = int(rng.integers(low=0, high=m))
    a = int(a)
    c = int(c)
    bit_length = np.ceil(np.log2(m)).astype(int)
    bits_to_drop = bit_length - bits_to_keep
    assert bits_to_keep > 0 and bits_to_keep <= bit_length, "Invalid bits_to_keep value"
    mask = (1 << bits_to_keep) - 1
    
    for i in range(num_examples * seq_len):
        x = (a * x + c) % m
        truncated_x = (x >> bits_to_drop) & mask
        array[i] = truncated_x

    reshaped_array = array.reshape(num_examples, seq_len)
    return reshaped_array


def base_tlcg(m: int = 512, seq_len: int = 8, a: int = 45, c: int = 123, bits_to_keep: int = 8, 
              base: int = 2, digits: int = 8, num_examples: int = 1, 
              rng: np.random.Generator = np.random.default_rng(97)) -> np.ndarray:
    """Truncated LCG with base-b representation"""
    return convert_to_base_b(
        truncated_lcg, base=base, digits=digits,
        m=m, seq_len=seq_len, a=a, c=c, bits_to_keep=bits_to_keep, 
        num_examples=num_examples, rng=rng
    )


def pcg_rs(m: int = 2**16, seq_len: int = 64, a: int = 45, c: int = 123, control_bits: int = 2, bits_to_keep: int = 8, rng: np.random.Generator = np.random.default_rng(97), num_examples: Optional[int] = None) -> np.ndarray:
    """PCG Random Shift"""
    if not num_examples:
        num_examples = m // seq_len

    x = int(rng.integers(low=0, high=m))
    a = int(a)
    c = int(c)
    array = np.zeros(num_examples * seq_len)
    bit_length = np.ceil(np.log2(m)).astype(int)
    mask = (1 << bits_to_keep) - 1
    assert bit_length > (2 ** control_bits - 1) + bits_to_keep, "need more bits"
    top_shift = bit_length - control_bits
    for i in range(num_examples * seq_len):
        x = (a * x + c) % m
        shift = bit_length - bits_to_keep - control_bits - (x >> top_shift)
        permuted_x = (x >> shift) & mask
        array[i] = permuted_x
    reshaped_array = array.reshape(num_examples, seq_len)
    
    return reshaped_array


def rotate_right(n: int, bits: int, width: int = 32) -> int:
    """Rotate the bits of n to the right by the specified number of bits"""
    bits = bits % width
    return ((n >> bits) & (2**width - 1)) | (n << (width - bits) & (2**width - 1))


def rotate_left(n: int, bits: int, width: int = 32) -> int:
    """Rotate the bits of n to the left by the specified number of bits"""
    bits = bits % width
    return ((n << bits) & (2**width - 1)) | (n >> (width - bits))


def pcg_rr(m: int = 2**16, seq_len: int = 64, a: int = 45, c: int = 123, control_bits: int = 3, bits_to_keep: int = 8, rng: np.random.Generator = np.random.default_rng(97), num_examples: Optional[int] = None) -> np.ndarray:
    """PCG Random Rotation"""
    if not num_examples:
        num_examples = m // seq_len

    array = np.zeros(num_examples * seq_len)
    x = int(rng.integers(low=0, high=m))
    a = int(a)
    c = int(c)
    bit_length = np.ceil(np.log2(m)).astype(int)
    mask = (1 << bits_to_keep) - 1
    assert bits_to_keep >= 2 ** control_bits, "cant do the permutation, need more bits"
    top_shift = bit_length - control_bits
    keep_shift = bit_length - control_bits - bits_to_keep
    for i in range(num_examples * seq_len):
        x = (a * x + c) % m
        rotation = (x >> top_shift)
        bits_kept = (x >> keep_shift) & mask
        rotated_x = rotate_right(bits_kept, rotation, width=bits_to_keep)
        array[i] = rotated_x
    reshaped_array = array.reshape(num_examples, seq_len)

    return reshaped_array


def pcg_xsh_rr(m: int = 2**16, seq_len: int = 64, a: int = 49, c: int = 123, control_bits: int = 3, bits_to_keep: int = 8, rng: np.random.Generator = np.random.default_rng(97), num_examples: Optional[int] = None) -> np.ndarray:
    """PCG XSH-RR (XOR Shift High - Random Rotation)"""
    if num_examples:
        n = min(m // seq_len, num_examples)
    else:
        n = m // seq_len

    array = np.zeros(n * seq_len, dtype=np.int64)
    x = int(rng.integers(low=0, high=m))
    a = int(a)
    c = int(c)
    bit_length = np.ceil(np.log2(m)).astype(int)
    assert bit_length > control_bits + bits_to_keep, "cant do the permutation, need more bits"
    shift = bits_to_keep - control_bits
    xor_shift = int((bits_to_keep + control_bits)/2)
    rotation_control = bit_length - control_bits
    mask = (1 << bits_to_keep) - 1

    for i in range(n * seq_len):
        x = (a * x + c) % m
        target = ((x ^ (x >> xor_shift)) >> shift) & mask
        rotation = (x >> rotation_control) 
        array[i] = rotate_right(target, rotation, width=bits_to_keep)

    reshaped_array = array.reshape(n, seq_len)
    return reshaped_array


def pcg_xsh_rs(m: int = 2**16, seq_len: int = 64, a: int = 49, c: int = 123, control_bits: int = 3, bits_to_keep: int = 8, rng: np.random.Generator = np.random.default_rng(97), num_examples: Optional[int] = None) -> np.ndarray:
    """PCG XSH-RS (XOR Shift High - Random Shift)"""
    if num_examples:
        n = min(m // seq_len, num_examples)
    else:
        n = m // seq_len

    array = np.zeros(n * seq_len, dtype=np.int64)
    x = int(rng.integers(low=0, high=m))
    a = int(a)
    c = int(c)
 
    bit_length = np.ceil(np.log2(m)).astype(int)
    control_shift = bit_length - control_bits
    mask = (1 << bits_to_keep) - 1
    constant_shift = bits_to_keep - control_bits - 2 ** control_bits + 1
    assert constant_shift > 0, "too many control bits for this state length"
    for i in range(n * seq_len):
        x = (a * x + c) % m
        permuted_x = (x ^ (x >> constant_shift)) >> (constant_shift + (x >> control_shift)) 
        array[i] = permuted_x & mask

    reshaped_array = array.reshape(n, seq_len)
    return reshaped_array


def pcg_xsl_rr(m: int = 2**16, seq_len: int = 64, a: int = 49, c: int = 123, control_bits: int = 3, num_examples: Optional[int] = None, rng: np.random.Generator = np.random.default_rng(97)) -> np.ndarray:
    """PCG XSL-RR (XOR Shift Low - Random Rotation)"""
    if num_examples:
        n = min(m // seq_len, num_examples)
    else:
        n = m // seq_len

    array = np.zeros(n * seq_len, dtype=np.int64)
    x = int(rng.integers(low=0, high=m))
    a = int(a)
    c = int(c)
    bit_length = np.ceil(np.log2(m)).astype(int)
 
    control_shift = bit_length - control_bits
    bits_to_keep = int(bit_length / 2)
    mask = (1 << bits_to_keep) - 1

    for i in range(n * seq_len):
        x = (a * x + c) % m
        xor_shifted_x = (x ^ (x >> bits_to_keep)) & mask
        array[i] = rotate_right(xor_shifted_x, x >> control_shift, width=bits_to_keep)

    reshaped_array = array.reshape(n, seq_len)
    return reshaped_array



def decimal_to_base_b_reverse(n: int, b: int, length: int) -> list:
    """Convert decimal number to base-b representation (kept for backward compatibility)"""
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    while len(digits) < length:
        digits.append(0)
    return digits


# Cache for divisor arrays to avoid recomputing powers
_divisor_cache = {}

def vectorized_base_conversion(values: np.ndarray, base: int, digits: int) -> np.ndarray:
    """

    Args:
        values: 1D array of decimal values to convert
        base: Base for the number system
        digits: Number of digits per number
    
    Returns:
        2D array of shape (len(values), digits) with base-b representation
    """
    # Special optimization for power-of-2 bases using bit operations
    if base > 1 and (base & (base - 1)) == 0:  # Check if base is power of 2
        bits_per_digit = int(np.log2(base))
        mask = base - 1  # Create bit mask (e.g., for base=8, mask=7=0b111)
        
        values_expanded = values[:, np.newaxis]
        shift_amounts = np.arange(digits) * bits_per_digit
        result = (values_expanded >> shift_amounts) & mask
        return result
    
    # General case: use cached divisors for arbitrary bases
    cache_key = (base, digits)
    if cache_key not in _divisor_cache:
        # Create divisor array: [1, base, base^2, base^3, ...]
        _divisor_cache[cache_key] = np.power(base, np.arange(digits, dtype=np.int64))
    
    divisors = _divisor_cache[cache_key]
    
    # Broadcast values to shape (n_values, digits) and apply divisors
    values_expanded = values[:, np.newaxis]  # Shape: (n_values, 1)
    
    # Vectorized computation: extract each digit position simultaneously
    # For each position i, we want (values // base^i) % base
    result = (values_expanded // divisors) % base
    
    return result


def convert_to_base_b(prng_func, base: int, digits: int, **kwargs) -> np.ndarray:
    """
    Universal wrapper to convert any PRNG function output to base-b representation.
    Args:
        prng_func: The base PRNG function to call
        base: Base for the number system
        digits: Number of digits per number
        **kwargs: All other parameters passed to the PRNG function
    
    Returns:
        np.ndarray with base-b representation
    """
    # Early return for regular representation - avoid any unnecessary computation
    # Always return a NumPy array for downstream efficiency
    if digits == 1:
        result = prng_func(**kwargs)
        return np.asarray(result, dtype=np.int64)
    
    # Call the original PRNG function to get regular output
    result = prng_func(**kwargs)
    
    # Work directly with numpy for efficiency - avoid copying when possible
    array = np.asarray(result, dtype=np.int64)
    
    # Get dimensions and flatten in one step for better cache efficiency
    num_examples, length = array.shape
    flat_values = array.ravel()  # This is a view, not a copy
    
    # Vectorized base conversion - the core optimization
    base_b_matrix = vectorized_base_conversion(flat_values, base, digits)
    

    result_array = base_b_matrix.reshape(num_examples, length * digits)
    
    # Return NumPy array; dataset assembly will convert once to torch
    return result_array


def base_b_pcg_rs(m: int = 2**16, seq_len: int = 64, a: int = 45, c: int = 123, 
                  control_bits: int = 2, bits_to_keep: int = 8, base: int = 256, digits: int = 1,
                  rng: np.random.Generator = np.random.default_rng(97), num_examples: Optional[int] = None) -> np.ndarray:
    """PCG RS with base-b representation"""
    return convert_to_base_b(
        pcg_rs, base=base, digits=digits,
        m=m, seq_len=seq_len, a=a, c=c, control_bits=control_bits,
        bits_to_keep=bits_to_keep, rng=rng, num_examples=num_examples
    )


def base_b_pcg_rr(m: int = 2**16, seq_len: int = 64, a: int = 45, c: int = 123, 
                  control_bits: int = 3, bits_to_keep: int = 8, base: int = 256, digits: int = 1,
                  rng: np.random.Generator = np.random.default_rng(97), num_examples: Optional[int] = None) -> np.ndarray:
    """PCG RR with base-b representation"""
    return convert_to_base_b(
        pcg_rr, base=base, digits=digits,
        m=m, seq_len=seq_len, a=a, c=c, control_bits=control_bits,
        bits_to_keep=bits_to_keep, rng=rng, num_examples=num_examples
    )


def base_b_pcg_xsh_rs(m: int = 2**16, seq_len: int = 64, a: int = 49, c: int = 123, 
                      control_bits: int = 3, bits_to_keep: int = 8, base: int = 256, digits: int = 1,
                      rng: np.random.Generator = np.random.default_rng(97), num_examples: Optional[int] = None) -> np.ndarray:
    """PCG XSH-RS with base-b representation"""
    return convert_to_base_b(
        pcg_xsh_rs, base=base, digits=digits,
        m=m, seq_len=seq_len, a=a, c=c, control_bits=control_bits,
        bits_to_keep=bits_to_keep, rng=rng, num_examples=num_examples
    )


def base_b_pcg_xsl_rr(m: int = 2**16, seq_len: int = 64, a: int = 49, c: int = 123, 
                      control_bits: int = 3, bits_to_keep: int = 8, base: int = 256, digits: int = 1, 
                      num_examples: Optional[int] = None, rng: np.random.Generator = np.random.default_rng(97)) -> np.ndarray:
    """PCG XSL-RR with base-b representation"""
    # Validate that bits_to_keep is half the bit length
    bit_length = int(np.ceil(np.log2(m)))
    expected_bits_to_keep = int(bit_length / 2)
    if bits_to_keep != expected_bits_to_keep:
        print(f"Warning: bits_to_keep={bits_to_keep} should be {expected_bits_to_keep} (half of bit_length={bit_length}). Using {expected_bits_to_keep}.")
    
    return convert_to_base_b(
        pcg_xsl_rr, base=base, digits=digits,
        m=m, seq_len=seq_len, a=a, c=c, control_bits=control_bits,
        num_examples=num_examples, rng=rng
    )


def base_b_pcg_xsh_rr(m: int = 2**16, seq_len: int = 64, a: int = 49, c: int = 123, 
                      control_bits: int = 3, bits_to_keep: int = 8, base: int = 256, digits: int = 1,
                      rng: np.random.Generator = np.random.default_rng(97), num_examples: Optional[int] = None) -> np.ndarray:
    """PCG XSH-RR with base-b representation"""
    return convert_to_base_b(
        pcg_xsh_rr, base=base, digits=digits,
        m=m, seq_len=seq_len, a=a, c=c, control_bits=control_bits,
        bits_to_keep=bits_to_keep, rng=rng, num_examples=num_examples
    )



