# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Linux syscall allowlist for the disposable, in-memory dependency worker.

Install only after trusted scanner imports. No paths, sockets, processes, or
new descriptors can be opened afterwards. Unsupported kernels fail closed.
This is deliberately separate from the host scanner and its network client.
"""

from __future__ import annotations

import ctypes
import errno
import platform
import sys


def restrict_worker() -> None:
    """Deny filesystem access, networking, process creation and execution."""
    if sys.platform != "linux":
        raise OSError("dependency worker sandbox requires Linux")
    machine = platform.machine()
    if machine == "x86_64":
        architecture = 0xC000003E
        seccomp_syscall = 317
        allowed = (
            0,
            1,
            3,
            5,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            17,
            19,
            20,
            24,
            25,
            28,
            39,
            60,
            186,
            202,
            204,
            228,
            230,
            231,
            318,
        )
    elif machine in {"aarch64", "arm64"}:
        architecture = 0xC00000B7
        seccomp_syscall = 277
        allowed = (
            57,
            62,
            63,
            64,
            65,
            66,
            67,
            80,
            93,
            94,
            98,
            113,
            115,
            123,
            124,
            134,
            135,
            139,
            172,
            178,
            214,
            215,
            216,
            222,
            226,
            233,
            278,
        )
    else:
        raise OSError("unsupported dependency worker architecture")

    class Filter(ctypes.Structure):
        _fields_ = [
            ("code", ctypes.c_ushort),
            ("jt", ctypes.c_ubyte),
            ("jf", ctypes.c_ubyte),
            ("k", ctypes.c_uint32),
        ]

    class Program(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ushort), ("filter", ctypes.POINTER(Filter))]

    # Verify audit architecture, then dispatch on the syscall number. In
    # particular x32 syscall numbers cannot match the x86-64 allowlist.
    instructions = [
        (0x20, 0, 0, 4),
        (0x15, 1, 0, architecture),
        (0x06, 0, 0, 0x80000000),
        (0x20, 0, 0, 0),
    ]
    for number in allowed:
        instructions.extend(((0x15, 0, 1, number), (0x06, 0, 0, 0x7FFF0000)))
    instructions.append((0x06, 0, 0, 0x00050000 | errno.EPERM))
    filters = (Filter * len(instructions))(*(Filter(*item) for item in instructions))
    program = Program(len(instructions), filters)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        raise OSError(ctypes.get_errno(), "cannot restrict dependency worker")
    # SECCOMP_SET_MODE_FILTER + SECCOMP_FILTER_FLAG_TSYNC also restrict any
    # threads started by trusted imports. Failure to synchronize fails closed.
    if libc.syscall(seccomp_syscall, 1, 1, ctypes.byref(program)) != 0:
        raise OSError(ctypes.get_errno(), "cannot sandbox dependency worker")
