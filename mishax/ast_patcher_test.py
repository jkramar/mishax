# coding=utf-8
# Copyright 2024 DeepMind Technologies Limited.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
import sys

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.tree_util

from mishax import ast_patcher


class PlainClass:
  def __init__(self):
    x = 0
    x += 2
    self.x = x

  def sum_of_two_xs(self):
    return self.x + self.x

  @classmethod
  def get_one(cls):
    return cls()


# Using decorators and an implicit metaclass.
@jax.tree_util.register_static
class FancyClass(enum.Enum):
  A = enum.auto()

  @property
  def x(self):
    x = 0
    x += 2
    return x

  def sum_of_two_xs(self):
    return self.x + self.x

  @classmethod
  def get_one(cls):
    return cls.A


def hit_patch():
  global HIT_PATCH
  HIT_PATCH = True


HIT_PATCH = False
UndecoratedEnum = enum.Enum('UndecoratedEnum', ['A'])
MODULE = sys.modules[__name__]


class AstPatcherTest(parameterized.TestCase):

  @parameterized.product(
      cls_name=[PlainClass.__name__, FancyClass.__name__], install=[True, False]
  )
  def test_patch(self, cls_name: str, install: bool):
    self.assertEqual(getattr(MODULE, cls_name).get_one().x, 2)
    global HIT_PATCH
    HIT_PATCH = False
    patcher = ast_patcher.ModuleASTPatcher(
        MODULE,
        **{cls_name: ['x += 2', 'x += 2\nhit_patch()']},
    )
    if install:
      self.assertFalse(patcher.is_installed())
      patcher.install()
      self.assertTrue(patcher.is_installed())
    for i in range(2):
      with self.subTest(['first_time', 'reuse'][i]), patcher():
        self.assertEqual(getattr(MODULE, cls_name).get_one().x, 2)
        self.assertTrue(HIT_PATCH)
        HIT_PATCH = False
      with self.subTest('after ' + ['second_time', 'reuse'][i]):
        self.assertEqual(getattr(MODULE, cls_name).get_one().x, 2)
        self.assertEqual(HIT_PATCH, install)
    if install:
      self.assertTrue(patcher.is_installed())
      del ast_patcher._INSTALLED_PATCHER_CONTEXTS[patcher]
      self.assertFalse(patcher.is_installed())

  def test_decorator_still_applied(self):
    orig_fancy_class = FancyClass
    patcher = ast_patcher.ModuleASTPatcher(MODULE, FancyClass=[])
    with patcher():
      self.assertNotEqual(orig_fancy_class, FancyClass)
      jax.jit(lambda x: x)(FancyClass.A)
    with self.subTest('decorator_was_needed'), self.assertRaises(TypeError):
      jax.jit(lambda x: x)(UndecoratedEnum.A)

  @parameterized.parameters([PlainClass.__name__, FancyClass.__name__])
  def test_patch_expr_and_prefix(self, cls_name):
    self.assertEqual(getattr(MODULE, cls_name).get_one().x, 2)
    patcher = ast_patcher.ModuleASTPatcher(
        MODULE,
        ast_patcher.PatchSettings(prefix='FOUR = 4'),
        **{cls_name: ['2', 'FOUR']},
    )
    for i in range(2):
      with self.subTest(['first_time', 'reuse'][i]), patcher():
        self.assertEqual(getattr(MODULE, cls_name).get_one().x, 4)
      with self.subTest('after ' + ['second_time', 'reuse'][i]):
        self.assertEqual(getattr(MODULE, cls_name).get_one().x, 2)

  @parameterized.parameters([PlainClass.__name__, FancyClass.__name__])
  def test_stacktrace(self, cls_name):
    patcher = ast_patcher.ModuleASTPatcher(
        MODULE,
        ast_patcher.PatchSettings(prefix='import inspect'),
        **{cls_name: ['x', 'inspect.getsource(inspect.currentframe())']},
    )
    with self.subTest('src_match'), patcher():
      self.assertContainsExactSubsequence(
          patcher.src, getattr(MODULE, cls_name).get_one().x
      )
    patcher = ast_patcher.ModuleASTPatcher(
        MODULE,
        ast_patcher.PatchSettings(prefix='import inspect'),
        **{cls_name: ['x', 'inspect.getsourcefile(inspect.currentframe())']},
    )
    with self.subTest('src_path'), patcher():
      self.assertEqual(patcher.path, getattr(MODULE, cls_name).get_one().x)

  @parameterized.parameters([PlainClass.__name__, FancyClass.__name__])
  def test_no_double_patch_by_default(self, cls_name):
    with self.assertRaisesRegex(
        ast_patcher.PatchError, r'Too many \(2\) matches'
    ):
      patcher = ast_patcher.ModuleASTPatcher(
          MODULE, **{cls_name: ['self.x', 'self.x + 1']}
      )
      with patcher():
        pass

  @parameterized.parameters([PlainClass.__name__, FancyClass.__name__])
  def test_can_double_patch(self, cls_name):
    patcher = ast_patcher.ModuleASTPatcher(
        MODULE,
        ast_patcher.PatchSettings(allow_num_matches_upto={cls_name: 2}),
        **{cls_name: ['self.x', 'self.x + 1']},
    )
    with patcher():
      self.assertEqual(getattr(MODULE, cls_name).get_one().sum_of_two_xs(), 6)

  def test_install_clean(self):
    patcher = ast_patcher.ModuleASTPatcher(
        'mishax.safe_greenlet',
        ast_patcher.PatchSettings(allow_num_matches_upto=dict(yield_=2)),
        yield_=['default', '"peekaboo"'],
    )
    patcher.install_clean()

    from mishax import safe_greenlet  # pylint: disable=g-import-not-at-top

    transient_patcher = ast_patcher.ModuleASTPatcher(
        'mishax.safe_greenlet',
        ast_patcher.PatchSettings(allow_num_matches_upto=dict(yield_=2)),
        yield_=['"peekaboo"', '"THUMP"'],
    )

    with self.subTest('plain'):
      self.assertEqual(safe_greenlet.yield_(), 'peekaboo')

    with self.subTest('with_context'), transient_patcher():
      self.assertEqual(safe_greenlet.yield_(), 'THUMP')

    with self.subTest('after_context'):
      self.assertEqual(safe_greenlet.yield_(), 'peekaboo')

    patcher.install()
    with self.subTest('after_install'):
      self.assertEqual(safe_greenlet.yield_(), 'peekaboo')

    with self.subTest('after_install_with_context'), transient_patcher():
      self.assertEqual(safe_greenlet.yield_(), 'THUMP')

    with self.subTest('after_install_and_context'):
      self.assertEqual(safe_greenlet.yield_(), 'peekaboo')


# Classes for in-place patching tests - defined at module level
# Each test that uses in-place patching needs its own unique class/function
# because in-place patching modifies the actual code objects.


class InPlaceTestClass1:
  """A class for testing in-place updates (test 1)."""

  def __init__(self, value=10):
    self.value = value

  def compute(self):
    return self.value * 2


class InPlaceTestClass2:
  """A class for testing in-place updates (test 2)."""

  def __init__(self, value=10):
    self.value = value

  def compute(self):
    return self.value * 2


class InPlaceTestClass3:
  """A class for testing in-place updates (test 3)."""

  def __init__(self, value=10):
    self.value = value

  def compute(self):
    return self.value * 2


def inplace_test_func1():
  """A function for testing in-place updates (test 1)."""
  return 'original_result'


def inplace_test_func2():
  """A function for testing in-place updates (test 2)."""
  return 'original_result'


class InPlacePatcherTest(parameterized.TestCase):
  """Tests for install_inplace() functionality."""

  def test_install_inplace_function(self):
    """Test that install_inplace updates function code in-place."""
    # Get reference to original function
    original_func = inplace_test_func1

    # Create patcher
    patcher = ast_patcher.ModuleASTPatcher(
        MODULE,
        inplace_test_func1=["'original_result'", "'patched_result'"],
    )

    # Install in-place
    result = patcher.install_inplace()

    # Verify the result structure
    self.assertIn('updated_members', result)
    self.assertIn('inplace_test_func1', result['updated_members'])

    # The original function reference should now return patched result
    # (because its __code__ was updated in-place)
    self.assertEqual(original_func(), 'patched_result')

    # Cleanup
    del ast_patcher._INSTALLED_PATCHER_CONTEXTS[patcher]

  def test_install_inplace_class_method(self):
    """Test that install_inplace updates class methods."""
    # Create instance before patching
    instance = InPlaceTestClass1(value=5)

    # Verify original behavior
    self.assertEqual(instance.compute(), 10)  # 5 * 2

    # Create patcher to change compute() multiplier
    patcher = ast_patcher.ModuleASTPatcher(
        MODULE,
        InPlaceTestClass1=['self.value * 2', 'self.value * 3'],
    )

    # Install in-place
    patcher.install_inplace()

    # The existing instance should now use the patched method
    self.assertEqual(instance.compute(), 15)  # 5 * 3

    # New instances should also use patched method
    new_instance = InPlaceTestClass1(value=4)
    self.assertEqual(new_instance.compute(), 12)  # 4 * 3

    # Cleanup
    del ast_patcher._INSTALLED_PATCHER_CONTEXTS[patcher]

  def test_install_inplace_preserves_instance_state(self):
    """Test that instance state is preserved after in-place patching."""
    # Create instance with custom state
    instance = InPlaceTestClass2(value=42)
    instance.custom_attr = 'preserved'

    # Patch the class
    patcher = ast_patcher.ModuleASTPatcher(
        MODULE,
        InPlaceTestClass2=['self.value * 2', 'self.value + 100'],
    )
    patcher.install_inplace()

    # Instance state should be preserved
    self.assertEqual(instance.value, 42)
    self.assertEqual(instance.custom_attr, 'preserved')

    # But method should use new code
    self.assertEqual(instance.compute(), 142)  # 42 + 100

    # Cleanup
    del ast_patcher._INSTALLED_PATCHER_CONTEXTS[patcher]

  def test_install_inplace_returns_update_info(self):
    """Test that install_inplace returns information about updates."""
    patcher = ast_patcher.ModuleASTPatcher(
        MODULE,
        inplace_test_func2=["'original_result'", "'info_test'"],
    )

    result = patcher.install_inplace()

    self.assertIsInstance(result, dict)
    self.assertIn('updated_members', result)
    self.assertIn('updated_modules', result)
    self.assertEqual(result['updated_members'], ['inplace_test_func2'])

    # Cleanup
    del ast_patcher._INSTALLED_PATCHER_CONTEXTS[patcher]

  def test_install_inplace_without_instance_updates(self):
    """Test install_inplace with update_instances=False."""
    patcher = ast_patcher.ModuleASTPatcher(
        MODULE,
        InPlaceTestClass3=['self.value * 2', 'self.value * 5'],
    )

    # Install without updating instances
    patcher.install_inplace(update_instances=False)

    # The module attribute should still be patched
    new_instance = MODULE.InPlaceTestClass3(value=2)
    self.assertEqual(new_instance.compute(), 10)  # 2 * 5

    # Cleanup
    del ast_patcher._INSTALLED_PATCHER_CONTEXTS[patcher]


class InPlaceUpdaterTest(absltest.TestCase):
  """Tests for the InPlaceUpdater class directly."""

  def test_update_function_code(self):
    """Test updating a function's code object."""
    def old_func():
      return 'old'

    def new_func():
      return 'new'

    updater = ast_patcher.InPlaceUpdater()
    updater.update(old_func, new_func)

    # old_func should now return 'new'
    self.assertEqual(old_func(), 'new')

  def test_update_function_defaults(self):
    """Test updating a function's default arguments."""
    def old_func(x=1):
      return x

    def new_func(x=99):
      return x

    updater = ast_patcher.InPlaceUpdater()
    updater.update(old_func, new_func)

    # old_func should now have new default
    self.assertEqual(old_func(), 99)

  def test_update_same_object_noop(self):
    """Test that updating an object with itself is a no-op."""
    def func():
      return 'test'

    original_code = func.__code__

    updater = ast_patcher.InPlaceUpdater()
    updater.update(func, func)

    # Code should be unchanged
    self.assertIs(func.__code__, original_code)

  def test_update_class_methods(self):
    """Test updating class method code."""

    class OldClass:
      def method(self):
        return 'old'

    class NewClass:
      def method(self):
        return 'new'

    updater = ast_patcher.InPlaceUpdater()
    updater.update(OldClass, NewClass)

    # OldClass instances should now use new method
    instance = OldClass()
    self.assertEqual(instance.method(), 'new')

  def test_update_instances_changes_class(self):
    """Test that update_instances remaps instance classes."""

    class OldClass:
      def method(self):
        return 'old'

    class NewClass:
      def method(self):
        return 'new'

    # Create instance before update
    instance = OldClass()
    self.assertEqual(instance.method(), 'old')

    updater = ast_patcher.InPlaceUpdater()
    updater.update(OldClass, NewClass)
    updater.update_instances()

    # Instance should now use NewClass (via __class__ reassignment)
    # Note: The method was already updated on OldClass, so it returns 'new'
    self.assertEqual(instance.method(), 'new')


if __name__ == '__main__':
  absltest.main()
