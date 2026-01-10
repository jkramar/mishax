"""Cookbook of Code Modification Techniques

This module demonstrates different code modification scenarios and shows when to use:
1. AST Patcher (ModuleASTPatcher)
2. unittest.mock.patch
3. Both combined
4. Neither (alternatives needed)

Each example is self-contained and tested in cookbook_test.py.
"""

from __future__ import annotations

# Example modules to patch (defined here for testing)
# These simulate external libraries we want to modify


class SimpleCalculator:
    """Example: Simple arithmetic operations"""

    def add(self, a: int, b: int) -> int:
        return a + b

    def multiply(self, a: int, b: int) -> int:
        return a * b

    def complex_calculation(self, x: int) -> int:
        temp = x + 2
        temp = temp * 3
        return temp - 1


class DataProcessor:
    """Example: Data processing with side effects"""

    def __init__(self):
        self.results = []

    def process_item(self, item: str) -> str:
        result = item.upper()
        self.results.append(result)
        return result

    def process_batch(self, items: list[str]) -> list[str]:
        return [self.process_item(item) for item in items]


class NetworkClient:
    """Example: External dependencies"""

    def fetch_data(self, url: str) -> dict:
        # In reality, this would make a network call
        return {"url": url, "status": "success"}

    def save_data(self, data: dict) -> bool:
        # In reality, this would save to database
        return True


class ConfigurableService:
    """Example: Service with configuration"""

    def __init__(self, config: dict):
        self.config = config

    def get_setting(self, key: str) -> str:
        return self.config.get(key, "default")

    def process_with_setting(self, data: str, setting_key: str) -> str:
        setting = self.get_setting(setting_key)
        return f"{data}_{setting}"


class MLModel:
    """Example: Machine learning model (like transformers)"""

    def __init__(self, hidden_size: int = 128):
        self.hidden_size = hidden_size
        self.layer_count = 0

    def forward_layer(self, x: list[float]) -> list[float]:
        # Simulate a layer computation
        self.layer_count += 1
        return [v * 2.0 for v in x]

    def forward(self, x: list[float]) -> list[float]:
        h1 = self.forward_layer(x)
        h2 = self.forward_layer(h1)
        output = [v + 1.0 for v in h2]
        return output


def standalone_function(x: int, y: int) -> int:
    """Example: Module-level function"""
    return x + y + 10


def function_with_multiple_returns(x: int) -> str:
    """Example: Function with control flow"""
    if x > 0:
        return "positive"
    elif x < 0:
        return "negative"
    else:
        return "zero"


def function_calling_others(x: int) -> int:
    """Example: Function composition"""
    temp = standalone_function(x, 5)
    return temp * 2


# Constants and module-level variables
DEFAULT_TIMEOUT = 30
API_VERSION = "v1"


# ============================================================================
# COOKBOOK CATEGORIES
# ============================================================================

"""
CATEGORY 1: EXPRESSION REPLACEMENT
===================================
Best tool: AST Patcher

Use AST Patcher when you want to change how a specific expression is computed
while preserving the surrounding code structure.

Examples:
- Replace a literal value with a variable
- Change a constant to a computation
- Intercept a return value
- Modify an intermediate calculation
"""


"""
CATEGORY 2: STATEMENT ADDITION
===============================
Best tool: AST Patcher

Use AST Patcher to add statements (logging, callbacks, instrumentation)
without changing the core logic.

Examples:
- Add logging before/after existing statements
- Add instrumentation callbacks
- Add debug assertions
- Add metric collection
"""


"""
CATEGORY 3: MULTI-LINE BLOCK REPLACEMENT
========================================
Best tool: AST Patcher

Use AST Patcher to replace entire code blocks while maintaining structure.

Examples:
- Replace algorithm implementations
- Change initialization logic
- Modify error handling blocks
- Update data transformations
"""


"""
CATEGORY 4: RETURN VALUE MOCKING
=================================
Best tool: unittest.mock.patch

Use unittest.mock.patch when you just need to control what a function returns,
especially for:
- External dependencies (network, filesystem, database)
- Testing error conditions
- Simplifying complex dependencies
- Avoiding side effects

Examples:
- Mock network calls
- Mock database queries
- Mock file I/O
- Mock external API calls
"""


"""
CATEGORY 5: BEHAVIOR TRACKING (SPY)
====================================
Best tool: AST Patcher OR unittest.mock.patch

Use AST Patcher when you need to:
- Track calls within a method (not at module boundaries)
- Preserve exact behavior while recording
- Track intermediate values (not just inputs/outputs)
- Work with JIT-compiled code (JAX, TensorFlow)

Use unittest.mock.patch when you need to:
- Track calls at module boundaries
- Record call arguments and return values
- Simple assertion counting
- Don't need to preserve implementation details

Examples:
- Count how many times a computation occurs (AST Patcher)
- Track attention weights in transformers (AST Patcher)
- Verify API was called with correct args (mock.patch)
- Check database was accessed (mock.patch)
"""


"""
CATEGORY 6: ADDING IMPORTS/GLOBALS
===================================
Best tool: AST Patcher (prefix parameter)

Use AST Patcher's prefix to add imports or global variables that your
patches need.

Examples:
- Import logging module
- Add global counters
- Define helper functions
- Import callbacks
"""


"""
CATEGORY 7: CONDITIONAL PATCHING
=================================
Best tool: AST Patcher + Python conditionals in patch

Use AST Patcher with conditional logic in the patched code.

Examples:
- Only log when flag is set
- Different behavior based on input
- Debug mode vs production mode
"""


"""
CATEGORY 8: MULTIPLE MODIFICATIONS TO SAME FUNCTION
===================================================
Best tool: AST Patcher (allow_num_matches_upto)

Use AST Patcher when you need to modify multiple locations within the
same function.

Examples:
- Instrument every layer in a neural network
- Log every return statement
- Add timing to every step
- Collect intermediate values at multiple points
"""


"""
CATEGORY 9: CONSTRUCTOR/INITIALIZATION MODIFICATION
===================================================
Best tool: AST Patcher OR unittest.mock.patch

Use AST Patcher when you need to:
- Modify how objects are initialized internally
- Add initialization logging
- Change default values based on complex logic

Use unittest.mock.patch when you need to:
- Replace entire object with a mock
- Provide test doubles
- Avoid expensive initialization

Examples:
- Add logging to __init__ (AST Patcher)
- Change default config values (AST Patcher)
- Replace service with mock (mock.patch)
"""


"""
CATEGORY 10: ATTRIBUTE ACCESS INTERCEPTION
===========================================
Best tool: AST Patcher OR unittest.mock.patch (PropertyMock)

Use AST Patcher when you need to:
- Intercept computed properties
- Add logging to getters
- Modify return values of properties

Use unittest.mock.patch with PropertyMock when you need to:
- Replace property values in tests
- Mock configuration properties

Examples:
- Log when config is accessed (AST Patcher)
- Return test config values (PropertyMock)
"""


"""
CATEGORY 11: EXCEPTION HANDLING MODIFICATION
============================================
Best tool: AST Patcher

Use AST Patcher to modify exception handling behavior.

Examples:
- Add logging to except blocks
- Change exception types
- Add retry logic
- Wrap exceptions with context
"""


"""
CATEGORY 12: LOOP INSTRUMENTATION
==================================
Best tool: AST Patcher

Use AST Patcher to instrument loop bodies.

Examples:
- Add progress tracking
- Collect intermediate results
- Add iteration callbacks
- Log every Nth iteration
"""


"""
CATEGORY 13: COMBINING AST PATCHER + MOCK
==========================================
Best tool: Both

Use both when you need AST Patcher's precision for internal modifications
and mock.patch for external dependencies.

Examples:
- Patch internal computation + mock external API
- Instrument model layers + mock data loading
- Add logging internally + mock network calls
"""


"""
CATEGORY 14: CANNOT USE EITHER (ALTERNATIVES NEEDED)
====================================================

Some modifications are difficult or impossible with AST Patcher or mock.patch:

1. RUNTIME-ONLY BEHAVIORS (use hooks/callbacks if available)
   - Event handlers
   - Async callbacks
   - Signal handlers

2. COMPILED CODE (use FFI, ctypes, or vendor solutions)
   - C extensions
   - Rust/Cython modules
   - Binary libraries

3. DYNAMIC/RUNTIME CODE GENERATION (intercept at generation time)
   - eval() results
   - exec() results
   - Dynamically created classes

4. METACLASS/DESCRIPTOR MAGIC (modify the metaclass/descriptor)
   - Complex descriptor protocols
   - Metaclass behaviors
   - __getattribute__ overrides

Alternatives:
- Monkey patching (setattr on modules/classes)
- Inheritance/composition
- Decorator wrapping
- Import hooks
- Debugger/profiler APIs
"""


# ============================================================================
# DECISION TREE
# ============================================================================

"""
DECISION TREE: Which tool to use?
==================================

START HERE
│
├─ Need to modify EXTERNAL dependency?
│  ├─ Just need to control return value? → unittest.mock.patch
│  └─ Need to modify internal behavior? → AST Patcher
│
├─ Need to modify INTERNAL computation?
│  ├─ Just replace one expression? → AST Patcher
│  ├─ Add logging/instrumentation? → AST Patcher
│  └─ Replace entire block? → AST Patcher
│
├─ Need to TRACK/SPY on behavior?
│  ├─ Track at module boundary? → unittest.mock.patch
│  └─ Track internal computations? → AST Patcher
│
├─ Need to ADD imports/globals? → AST Patcher (prefix)
│
├─ Need BOTH internal modification AND external mocking? → Both
│
└─ Working with compiled code/metaclasses? → Alternatives needed

QUICK REFERENCE:
- AST Patcher: Surgical source-level modifications, instrumentation
- unittest.mock.patch: Replace/spy on module boundaries
- Both: Complex scenarios with internal + external changes
- Neither: Compiled code, metaclasses, runtime-only behaviors
"""
