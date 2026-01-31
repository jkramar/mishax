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
import pickle
import sys

from absl.testing import absltest
from absl.testing import parameterized
import cloudpickle
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


class OverlayModuleTest(parameterized.TestCase):
  """Tests for overlay module functionality."""

  def tearDown(self):
    super().tearDown()
    # Clean up any overlay modules created during tests.
    for key in list(sys.modules.keys()):
      if key.startswith('test_overlay_'):
        del sys.modules[key]

  def test_overlay_module_created(self):
    """Test that overlay module is created and registered in sys.modules."""
    overlay_name = 'test_overlay_created'
    patcher = ast_patcher.ModuleASTPatcher(
        MODULE,
        overlay_name=overlay_name,
        PlainClass=['x += 2', 'x += 3'],
    )
    # Access updated_members to trigger _setup()
    _ = patcher.updated_members

    self.assertIn(overlay_name, sys.modules)
    self.assertIsNotNone(patcher.overlay_module)
    self.assertEqual(patcher.overlay_module, sys.modules[overlay_name])
    self.assertEqual(patcher.overlay_name, overlay_name)

  def test_overlay_module_contains_patched_members(self):
    """Test that overlay module contains the patched class."""
    overlay_name = 'test_overlay_members'
    patcher = ast_patcher.ModuleASTPatcher(
        MODULE,
        overlay_name=overlay_name,
        PlainClass=['x += 2', 'x += 3'],
    )
    _ = patcher.updated_members

    overlay = sys.modules[overlay_name]
    self.assertTrue(hasattr(overlay, 'PlainClass'))
    self.assertEqual(overlay.PlainClass, patcher.updated_members['PlainClass'])

  def test_patched_class_module_attribute(self):
    """Test that patched class has __module__ set to overlay name."""
    overlay_name = 'test_overlay_module_attr'
    patcher = ast_patcher.ModuleASTPatcher(
        MODULE,
        overlay_name=overlay_name,
        PlainClass=['x += 2', 'x += 3'],
    )
    patched_class = patcher.updated_members['PlainClass']

    self.assertEqual(patched_class.__module__, overlay_name)
    # Methods should also have __module__ set
    self.assertEqual(patched_class.__init__.__module__, overlay_name)
    self.assertEqual(patched_class.sum_of_two_xs.__module__, overlay_name)

  def test_patched_function_module_attribute(self):
    """Test that patched functions have __module__ set to overlay name."""
    overlay_name = 'test_overlay_func_module'
    patcher = ast_patcher.ModuleASTPatcher(
        MODULE,
        overlay_name=overlay_name,
        hit_patch=['global HIT_PATCH', 'pass'],
    )
    patched_func = patcher.updated_members['hit_patch']

    self.assertEqual(patched_func.__module__, overlay_name)

  def test_unpatched_class_module_unchanged(self):
    """Test that original class __module__ is not affected by patching."""
    overlay_name = 'test_overlay_unpatched'
    patcher = ast_patcher.ModuleASTPatcher(
        MODULE,
        overlay_name=overlay_name,
        PlainClass=['x += 2', 'x += 3'],
    )
    _ = patcher.updated_members

    # Original class should still have original __module__
    self.assertEqual(PlainClass.__module__, MODULE.__name__)

  @parameterized.parameters([pickle, cloudpickle])
  def test_pickle_patched_class_instance(self, pickle_module):
    """Test that patched class instances can be pickled and unpickled."""
    overlay_name = 'test_overlay_pickle_instance'
    patcher = ast_patcher.ModuleASTPatcher(
        MODULE,
        overlay_name=overlay_name,
        PlainClass=['x += 2', 'x += 3'],
    )
    patched_class = patcher.updated_members['PlainClass']
    instance = patched_class()

    # The instance should be picklable
    pickled = pickle_module.dumps(instance)
    unpickled = pickle_module.loads(pickled)

    # The unpickled instance should have the patched value
    self.assertEqual(unpickled.x, 3)  # 0 + 3 = 3 (patched from += 2 to += 3)
    self.assertEqual(type(unpickled).__module__, overlay_name)

  @parameterized.parameters([pickle, cloudpickle])
  def test_pickle_patched_class(self, pickle_module):
    """Test that patched class itself can be pickled and unpickled."""
    overlay_name = 'test_overlay_pickle_class'
    patcher = ast_patcher.ModuleASTPatcher(
        MODULE,
        overlay_name=overlay_name,
        PlainClass=['x += 2', 'x += 3'],
    )
    patched_class = patcher.updated_members['PlainClass']

    # The class should be picklable
    pickled = pickle_module.dumps(patched_class)
    unpickled_class = pickle_module.loads(pickled)

    # The unpickled class should be the patched version
    self.assertEqual(unpickled_class.__module__, overlay_name)
    instance = unpickled_class()
    self.assertEqual(instance.x, 3)

  def test_pickle_distinguishes_patched_from_unpatched(self):
    """Test that pickle distinguishes patched objects from unpatched ones."""
    overlay_name = 'test_overlay_distinguish'
    patcher = ast_patcher.ModuleASTPatcher(
        MODULE,
        overlay_name=overlay_name,
        PlainClass=['x += 2', 'x += 3'],
    )
    patched_class = patcher.updated_members['PlainClass']
    unpatched_class = PlainClass

    # Pickle both classes
    patched_pickled = cloudpickle.dumps(patched_class)
    unpatched_pickled = cloudpickle.dumps(unpatched_class)

    # Unpickle both
    patched_unpickled = cloudpickle.loads(patched_pickled)
    unpatched_unpickled = cloudpickle.loads(unpatched_pickled)

    # They should be different classes with different __module__ values
    self.assertEqual(patched_unpickled.__module__, overlay_name)
    self.assertEqual(unpatched_unpickled.__module__, MODULE.__name__)

    # And produce different results
    self.assertEqual(patched_unpickled().x, 3)
    self.assertEqual(unpatched_unpickled().x, 2)

  def test_no_overlay_when_not_specified(self):
    """Test that no overlay is created when overlay_name is not specified."""
    patcher = ast_patcher.ModuleASTPatcher(
        MODULE,
        PlainClass=['x += 2', 'x += 3'],
    )
    _ = patcher.updated_members

    self.assertIsNone(patcher.overlay_name)
    self.assertIsNone(patcher.overlay_module)

  def test_overlay_module_with_install(self):
    """Test that overlay module works with install()."""
    overlay_name = 'test_overlay_install'
    patcher = ast_patcher.ModuleASTPatcher(
        MODULE,
        overlay_name=overlay_name,
        PlainClass=['x += 2', 'x += 3'],
    )
    patcher.install()

    try:
      # Check the module is installed with patched version
      self.assertEqual(MODULE.PlainClass().x, 3)

      # Check overlay module exists
      self.assertIn(overlay_name, sys.modules)

      # Pickle and unpickle the class from the module
      pickled = cloudpickle.dumps(MODULE.PlainClass)
      unpickled = cloudpickle.loads(pickled)

      # Should be the patched version from the overlay
      self.assertEqual(unpickled.__module__, overlay_name)
      self.assertEqual(unpickled().x, 3)
    finally:
      # Clean up
      if patcher in ast_patcher._INSTALLED_PATCHER_CONTEXTS:
        del ast_patcher._INSTALLED_PATCHER_CONTEXTS[patcher]


if __name__ == '__main__':
  absltest.main()
